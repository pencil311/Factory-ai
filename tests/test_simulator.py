"""Tests for the physics-informed simulator and source registry.

No live Mongo: the simulator is fed the seed sensor catalogue directly, which is
pure data. Fault behaviour is asserted on *relationships between signals* — lag,
correlation, ordering — because that is what makes the simulator worth having.
A test that only checked "vibration went up" would pass on random noise.
"""

from __future__ import annotations

import pytest

from app.models.machine import SensorType
from app.models.reading import ReadingQuality, ReadingSource
from app.sensors import registry
from app.sensors.mqtt_source import MqttSensorSource, TopicBinding, topic_to_regex
from app.sensors.opcua_source import NodeBinding, quality_from_status_code
from app.sensors.simulator import FAULT_MODELS, FaultType, MachineSimulator
from app.seed.seed_machines import SENSORS

TENANT = "demo"
MACHINE = "CV-201"
VIBRATION = "CV-201-VIB-01"
TEMPERATURE = "CV-201-TMP-01"
RPM = "CV-201-RPM-01"


@pytest.fixture
def sim() -> MachineSimulator:
    """A connected simulator over the real seeded sensor catalogue."""
    simulator = MachineSimulator(
        tenant_id=TENANT,
        sensors=[s.model_dump() for s in SENSORS],
        interval_seconds=2.0,
    )
    simulator._connected = True  # connect() would otherwise hit Mongo
    return simulator


def value_of(sim: MachineSimulator, sensor_id: str) -> float:
    spec = sim.spec(sensor_id)
    return sim.compute_value(spec, sim.get_machine_state(spec.machine_id))


def run_hours(sim: MachineSimulator, hours: float, steps: int = 60) -> None:
    """Advance simulated time by ``hours``, in ``steps`` increments."""
    per_step = hours * 3600.0 / sim.time_scale / steps
    for _ in range(steps):
        sim.advance(per_step)


def fault_deltas(
    machine_id: str,
    fault_type: FaultType,
    sensor_ids: tuple[str, ...],
    hours: float = 500.0,
    step_hours: float = 1.0,
) -> list[tuple[float, dict[str, float]]]:
    """Isolate a fault's contribution by differencing against a healthy twin.

    Two simulators are advanced through identical simulated time; only one has
    the fault. Because the model is deterministic, ambient drift, production
    load and measurement noise are bit-identical in both and cancel on
    subtraction — what remains is exactly what the fault did.

    Without this, a daily temperature swing of 37->50 C swamps the early
    thermal signature and any threshold-based assertion just measures sunrise.

    Returns ``[(severity, {sensor_id: delta}), ...]`` per step.
    """
    catalogue = [s.model_dump() for s in SENSORS]
    healthy = MachineSimulator(tenant_id=TENANT, sensors=catalogue)
    faulted = MachineSimulator(tenant_id=TENANT, sensors=catalogue)
    for simulator in (healthy, faulted):
        simulator._connected = True
    faulted.inject_fault(machine_id, fault_type, severity=0.0)

    series: list[tuple[float, dict[str, float]]] = []
    for _ in range(int(hours / step_hours)):
        healthy.advance(step_hours * 3600.0)
        faulted.advance(step_hours * 3600.0)
        severity = faulted.get_machine_state(machine_id).faults[fault_type].severity
        series.append(
            (
                severity,
                {
                    sid: value_of(faulted, sid) - value_of(healthy, sid)
                    for sid in sensor_ids
                },
            )
        )
    return series


# ---------------------------------------------------------------------------
# Healthy behaviour
# ---------------------------------------------------------------------------
def test_healthy_machine_stays_within_normal_band(sim):
    """No fault, full health: every sensor stays inside normal_min..normal_max."""
    for _ in range(400):
        sim.advance(30.0)
        for spec in sim.sensor_specs:
            value = sim.compute_value(spec, sim.get_machine_state(spec.machine_id))
            assert spec.normal_min <= value <= spec.normal_max, (
                f"{spec.sensor_id} left its normal band at "
                f"{sim.get_machine_state(spec.machine_id).sim_seconds:.0f}s: "
                f"{value} not in [{spec.normal_min}, {spec.normal_max}]"
            )


def test_healthy_readings_are_not_flat(sim):
    """Load and diurnal drift must actually move the signal."""
    samples = []
    for _ in range(50):
        sim.advance(120.0)
        samples.append(value_of(sim, TEMPERATURE))

    assert len(set(samples)) > 40
    assert max(samples) - min(samples) > 0.5


def test_readings_are_deterministic_at_the_same_simulated_instant(sim):
    """Same instant, same values — replays and tests are reproducible."""
    sim.advance(1234.0)
    first = [value_of(sim, s.sensor_id) for s in sim.sensor_specs]
    second = [value_of(sim, s.sensor_id) for s in sim.sensor_specs]

    assert first == second


# ---------------------------------------------------------------------------
# Fault mechanics: ordering and lag
# ---------------------------------------------------------------------------
def test_bearing_wear_raises_vibration_before_temperature(sim):
    """The mechanism must precede its thermal consequence.

    Vibration comes from spalling; heat comes from the friction that spalling
    causes. A model where temperature moves first is physically backwards.
    """
    series = fault_deltas(MACHINE, FaultType.bearing_wear, (VIBRATION, TEMPERATURE))

    vib_moved_at = next(
        ((i, sev) for i, (sev, d) in enumerate(series) if d[VIBRATION] > 0.05), None
    )
    tmp_moved_at = next(
        ((i, sev) for i, (sev, d) in enumerate(series) if d[TEMPERATURE] > 0.05), None
    )

    assert vib_moved_at is not None, "vibration never responded to bearing wear"
    assert tmp_moved_at is not None, "temperature never followed the vibration"

    vib_step, vib_severity = vib_moved_at
    tmp_step, tmp_severity = tmp_moved_at
    assert vib_step < tmp_step, (
        f"temperature moved at step {tmp_step} before vibration at {vib_step}"
    )
    # Not a photo finish. The gap is the early-warning window the whole
    # pipeline exists to exploit: over a hundred hours of rising vibration
    # before the first degree of extra heat.
    assert tmp_step - vib_step > 100
    # The lag is structural, not incidental: temperature is pinned to zero
    # contribution until severity passes the model's lag.
    assert vib_severity < tmp_severity
    assert tmp_severity >= FAULT_MODELS[FaultType.bearing_wear].effects[
        SensorType.temperature
    ].lag


def test_temperature_cannot_move_before_its_lag_is_passed(sim):
    """Below the lag threshold, the thermal effect contributes exactly zero."""
    effect = FAULT_MODELS[FaultType.bearing_wear].effects[SensorType.temperature]
    assert effect.contribution(effect.lag) == 0.0
    assert effect.contribution(effect.lag - 0.01) == 0.0
    assert effect.contribution(effect.lag + 0.01) > 0.0
    assert effect.contribution(1.0) == pytest.approx(1.0)


def test_vibration_effect_is_exponential_not_linear():
    """Wear accelerates: the curve must be convex, not a straight ramp."""
    effect = FAULT_MODELS[FaultType.bearing_wear].effects[SensorType.vibration]
    quarter, half, three_quarter = (
        effect.contribution(0.25),
        effect.contribution(0.5),
        effect.contribution(0.75),
    )

    assert quarter < 0.25  # lags a linear ramp early
    assert half - quarter < three_quarter - half  # and accelerates later
    assert effect.contribution(1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Progression to failure
# ---------------------------------------------------------------------------
def test_severity_progression_crosses_warning_then_critical(sim):
    """Readings must pass through warning before reaching critical."""
    spec = sim.spec(VIBRATION)
    sim.inject_fault(MACHINE, FaultType.bearing_wear, severity=0.0)

    crossed_warning_at = None
    crossed_critical_at = None
    for _ in range(600):
        run_hours(sim, 1.0, steps=1)
        value = value_of(sim, VIBRATION)
        severity = sim.get_machine_state(MACHINE).faults[FaultType.bearing_wear].severity

        if crossed_warning_at is None and value >= spec.warning_threshold:
            crossed_warning_at = severity
        if value >= spec.critical_threshold:
            crossed_critical_at = severity
            break

    assert crossed_warning_at is not None, "vibration never reached the warning threshold"
    assert crossed_critical_at is not None, "vibration never reached the critical threshold"
    assert crossed_warning_at < crossed_critical_at


def test_health_falls_as_the_fault_develops(sim):
    sim.inject_fault(MACHINE, FaultType.bearing_wear, severity=0.0)
    assert sim.get_machine_state(MACHINE).health == pytest.approx(1.0)

    run_hours(sim, 300.0)
    mid_health = sim.get_machine_state(MACHINE).health
    assert 0.0 < mid_health < 1.0

    run_hours(sim, 400.0)
    assert sim.get_machine_state(MACHINE).health < mid_health


def test_fault_severity_saturates_at_failure(sim):
    sim.inject_fault(MACHINE, FaultType.bearing_wear, severity=0.5)
    run_hours(sim, 5000.0)

    assert sim.get_machine_state(MACHINE).faults[FaultType.bearing_wear].severity == 1.0
    assert sim.get_machine_state(MACHINE).health == 0.0


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
def test_clear_fault_returns_readings_toward_baseline(sim):
    healthy_vib = value_of(sim, VIBRATION)

    sim.inject_fault(MACHINE, FaultType.bearing_wear, severity=0.6)
    run_hours(sim, 100.0)
    faulted_vib = value_of(sim, VIBRATION)
    assert faulted_vib > healthy_vib * 2

    assert sim.clear_fault(MACHINE, FaultType.bearing_wear) is True
    repaired_vib = value_of(sim, VIBRATION)

    assert repaired_vib < faulted_vib
    spec = sim.spec(VIBRATION)
    assert spec.normal_min <= repaired_vib <= spec.normal_max
    assert sim.get_machine_state(MACHINE).health == pytest.approx(1.0)


def test_clear_fault_reports_when_nothing_was_wrong(sim):
    assert sim.clear_fault(MACHINE, FaultType.seal_leak) is False


def test_reset_restores_the_whole_fleet(sim):
    sim.inject_fault(MACHINE, FaultType.bearing_wear, severity=0.8)
    sim.inject_fault("MC-110", FaultType.tool_wear, severity=0.5)
    run_hours(sim, 50.0)

    sim.reset()

    for state in sim.get_all_states().values():
        assert state.faults == {}
        assert state.health == 1.0
        assert state.runtime_hours == 0.0
        assert state.sim_seconds == 0.0


# ---------------------------------------------------------------------------
# Correlation: signals must move together, plausibly
# ---------------------------------------------------------------------------
def test_temperature_rise_accompanies_vibration_rise_never_alone(sim):
    """Bearing wear: once heat appears, vibration is already elevated.

    This is the assertion that separates a physics model from six independent
    ramps — heat without vibration would mean the mechanism is missing.
    """
    series = fault_deltas(MACHINE, FaultType.bearing_wear, (VIBRATION, TEMPERATURE))

    hot_samples = 0
    for severity, delta in series:
        if delta[TEMPERATURE] > 0.05:
            hot_samples += 1
            assert delta[VIBRATION] > 0.5, (
                f"at severity {severity:.2f} the fault added "
                f"{delta[TEMPERATURE]:.2f} C of heat with only "
                f"{delta[VIBRATION]:.2f} mm/s of vibration behind it"
            )

    assert hot_samples > 50, "temperature never rose; the correlation went untested"

    # And the converse: there is a real window where vibration is elevated and
    # no heat has appeared yet. That asymmetry is the diagnostic signal.
    vibration_only = [
        s for s, d in series if d[VIBRATION] > 0.05 and d[TEMPERATURE] == 0.0
    ]
    assert len(vibration_only) > 50


def test_rpm_falls_while_vibration_and_temperature_rise(sim):
    """Bearing drag: speed must move opposite to vibration and heat."""
    before = {s: value_of(sim, s) for s in (VIBRATION, TEMPERATURE, RPM)}

    sim.inject_fault(MACHINE, FaultType.bearing_wear, severity=0.9)
    after = {s: value_of(sim, s) for s in (VIBRATION, TEMPERATURE, RPM)}

    assert after[VIBRATION] > before[VIBRATION]
    assert after[TEMPERATURE] > before[TEMPERATURE]
    assert after[RPM] < before[RPM]


def test_seal_leak_drops_pressure_while_power_rises(sim):
    """Different mechanism, different signature — pressure falls, not rises."""
    pressure_id, power_id = "HP-150-PRS-01", "HP-150-PWR-01"
    before_pressure = value_of(sim, pressure_id)
    before_power = value_of(sim, power_id)

    sim.inject_fault("HP-150", FaultType.seal_leak, severity=0.9)

    assert value_of(sim, pressure_id) < before_pressure
    assert value_of(sim, power_id) > before_power


def test_faults_on_unaffected_sensor_types_leave_them_alone(sim):
    """A seal leak has no vibration term; that sensor must not drift with it."""
    sim.advance(0.0)
    before = value_of(sim, "HP-150-VIB-01")

    sim.inject_fault("HP-150", FaultType.seal_leak, severity=1.0)

    assert value_of(sim, "HP-150-VIB-01") == pytest.approx(before)


# ---------------------------------------------------------------------------
# time_scale
# ---------------------------------------------------------------------------
def test_time_scale_speeds_progression_without_changing_values():
    """A 3-week failure replayed fast must pass through identical values.

    Both simulators are advanced to the same *simulated* instant by different
    wall-clock routes; every reading must match exactly, because the model is a
    function of simulated time alone.
    """
    catalogue = [s.model_dump() for s in SENSORS]

    real_time = MachineSimulator(tenant_id=TENANT, sensors=catalogue, time_scale=1.0)
    fast = MachineSimulator(tenant_id=TENANT, sensors=catalogue, time_scale=3600.0)
    for simulator in (real_time, fast):
        simulator._connected = True
        simulator.inject_fault(MACHINE, FaultType.bearing_wear, severity=0.2)

    # 100 simulated hours: 360,000 wall seconds vs 100. Same destination.
    for _ in range(100):
        real_time.advance(3600.0)
        fast.advance(1.0)

    real_state = real_time.get_machine_state(MACHINE)
    fast_state = fast.get_machine_state(MACHINE)

    assert fast_state.sim_seconds == pytest.approx(real_state.sim_seconds)
    assert fast_state.runtime_hours == pytest.approx(real_state.runtime_hours)
    assert fast_state.faults[FaultType.bearing_wear].severity == pytest.approx(
        real_state.faults[FaultType.bearing_wear].severity
    )
    assert fast_state.health == pytest.approx(real_state.health)

    for spec in real_time.sensor_specs:
        assert value_of(fast, spec.sensor_id) == pytest.approx(
            value_of(real_time, spec.sensor_id)
        ), f"{spec.sensor_id} diverged between real-time and accelerated replay"


def test_time_scale_reaches_failure_in_far_less_wall_time():
    """The demo case: three weeks of bearing wear inside two minutes."""
    catalogue = [s.model_dump() for s in SENSORS]
    # 504 simulated hours in 120 wall seconds.
    demo = MachineSimulator(
        tenant_id=TENANT, sensors=catalogue, time_scale=504 * 3600 / 120
    )
    demo._connected = True
    demo.inject_fault(MACHINE, FaultType.bearing_wear, severity=0.0)

    for _ in range(120):
        demo.advance(1.0)

    state = demo.get_machine_state(MACHINE)
    assert state.faults[FaultType.bearing_wear].severity == pytest.approx(1.0, abs=0.01)
    assert value_of(demo, VIBRATION) >= demo.spec(VIBRATION).critical_threshold


# ---------------------------------------------------------------------------
# Source interface
# ---------------------------------------------------------------------------
def test_tick_emits_a_reading_for_every_sensor(sim):
    readings = sim.tick()

    assert len(readings) == len(SENSORS)
    assert {r.sensor_id for r in readings} == {s.sensor_id for s in SENSORS}
    assert all(r.source == ReadingSource.simulator for r in readings)
    assert all(r.quality == ReadingQuality.good for r in readings)
    assert all(r.tenant_id == TENANT for r in readings)


@pytest.mark.asyncio
async def test_read_once_returns_one_sensor(sim):
    reading = await sim.read_once(VIBRATION)

    assert reading.sensor_id == VIBRATION
    assert reading.machine_id == MACHINE
    assert reading.unit == "mm/s"


@pytest.mark.asyncio
async def test_read_once_rejects_unknown_sensors(sim):
    with pytest.raises(KeyError):
        await sim.read_once("NO-SUCH-SENSOR")


def test_injecting_a_fault_on_an_unknown_machine_raises(sim):
    with pytest.raises(KeyError):
        sim.inject_fault("NOPE-999", FaultType.bearing_wear)


def test_set_health_degrades_readings_without_a_named_fault(sim):
    before = value_of(sim, VIBRATION)
    sim.set_health(MACHINE, 0.3)

    assert sim.get_machine_state(MACHINE).health == pytest.approx(0.3)
    assert value_of(sim, VIBRATION) > before


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the process-wide source out of the tests' way."""
    registry.reset_active_source()
    yield
    registry.reset_active_source()


def test_registry_defaults_to_simulator(monkeypatch):
    monkeypatch.delenv("SENSOR_SOURCE", raising=False)
    source = registry.build_source()

    assert isinstance(source, MachineSimulator)
    assert source.name == "simulator"
    assert source.source_type == ReadingSource.simulator


@pytest.mark.parametrize(
    "kind,expected_name",
    [("simulator", "simulator"), ("SIMULATOR", "simulator"), (" simulator ", "simulator")],
)
def test_registry_selects_simulator_by_env_var(kind, expected_name):
    assert registry.build_source(kind).name == expected_name


def test_registry_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="Unknown SENSOR_SOURCE"):
        registry.build_source("carrier-pigeon")


def test_registry_reports_missing_hardware_config():
    """opcua/mqtt need a mapping file; the error must say which one."""
    with pytest.raises(FileNotFoundError, match="node map"):
        registry.build_source("opcua")
    with pytest.raises(FileNotFoundError, match="topic map"):
        registry.build_source("mqtt")


def test_registry_returns_the_same_instance_every_call():
    """Injecting a fault must affect the source ingestion is reading."""
    assert registry.get_active_source() is registry.get_active_source()


def test_get_simulator_refuses_when_source_is_not_a_simulator():
    registry.set_active_source(
        MqttSensorSource(tenant_id=TENANT, host="localhost", bindings={})
    )
    with pytest.raises(LookupError, match="not 'simulator'"):
        registry.get_simulator()


# ---------------------------------------------------------------------------
# Hardware source normalisation (the parts that are not stubs)
# ---------------------------------------------------------------------------
def test_opcua_status_codes_map_onto_reading_quality():
    assert quality_from_status_code(0x00000000) == ReadingQuality.good
    assert quality_from_status_code(0x40000000) == ReadingQuality.suspect
    assert quality_from_status_code(0x80000000) == ReadingQuality.bad


def test_opcua_normalisation_applies_scale_and_offset():
    from app.sensors.opcua_source import OpcUaSensorSource

    binding = NodeBinding(
        sensor_id=TEMPERATURE,
        node_id="ns=2;s=CV201.Temp",
        machine_id=MACHINE,
        sensor_type=SensorType.temperature,
        unit="°C",
        scale=0.1,
        offset=-5.0,
    )
    source = OpcUaSensorSource(
        tenant_id=TENANT, endpoint="opc.tcp://x", bindings={TEMPERATURE: binding}
    )

    reading = source.normalize(binding, raw_value=615.0)

    assert reading.value == pytest.approx(56.5)
    assert reading.source == ReadingSource.opcua
    assert reading.quality == ReadingQuality.good


def test_mqtt_topic_wildcards_match_like_a_broker():
    assert topic_to_regex("factory/+/vibration").match("factory/CV-201/vibration")
    assert not topic_to_regex("factory/+/vibration").match("factory/a/b/vibration")
    assert topic_to_regex("factory/#").match("factory/a/b/c")


def test_mqtt_normalisation_reads_json_payloads():
    binding = TopicBinding(
        topic="factory/+/vib",
        sensor_id=VIBRATION,
        machine_id=MACHINE,
        sensor_type=SensorType.vibration,
        unit="mm/s",
    )
    source = MqttSensorSource(
        tenant_id=TENANT, host="localhost", bindings={"factory/+/vib": binding}
    )

    reading = source.normalize(
        "factory/CV-201/vib",
        b'{"value": 3.2, "timestamp": 1750000000, "quality": "UNCERTAIN"}',
    )

    assert reading.value == pytest.approx(3.2)
    assert reading.quality == ReadingQuality.suspect
    assert reading.source == ReadingSource.mqtt
    assert reading.timestamp.year == 2025


def test_mqtt_normalisation_accepts_bare_numeric_payloads():
    binding = TopicBinding(
        topic="factory/vib",
        sensor_id=VIBRATION,
        machine_id=MACHINE,
        sensor_type=SensorType.vibration,
        unit="mm/s",
    )
    source = MqttSensorSource(
        tenant_id=TENANT, host="localhost", bindings={"factory/vib": binding}
    )

    assert source.normalize("factory/vib", b"2.75").value == pytest.approx(2.75)


def test_mqtt_unmapped_topic_is_an_error_not_a_silent_drop():
    source = MqttSensorSource(tenant_id=TENANT, host="localhost", bindings={})

    with pytest.raises(KeyError, match="No sensor binding"):
        source.normalize("factory/unknown/vib", b"1.0")
