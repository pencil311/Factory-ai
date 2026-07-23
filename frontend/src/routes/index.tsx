import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Boxes,
  Brain,
  ChevronRight,
  Cpu,
  Factory,
  Gauge,
  Layers,
  LineChart,
  Lock,
  Play,
  ShieldCheck,
  Sparkles,
  Workflow,
  Zap,
} from "lucide-react";

import { BrandLogo } from "@/components/brand-logo";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Iron Sight — The Operating System for Modern Factories" },
      {
        name: "description",
        content:
          "Predictive maintenance, digital twins, and Elena, your AI copilot, in one enterprise command center. Trusted by industrial teams to cut downtime up to 42%.",
      },
      { property: "og:title", content: "Iron Sight — The Operating System for Modern Factories" },
      {
        property: "og:description",
        content: "Predictive maintenance, digital twins, and Elena, your AI copilot, in one enterprise command center. Trusted by industrial teams to cut downtime up to 42%.",
      },
    ],
  }),
  component: Landing,
});

const nav = [
  { label: "Product", href: "#product" },
  { label: "Features", href: "#features" },
  { label: "How it works", href: "#how" },
  { label: "Customers", href: "#customers" },
  { label: "FAQ", href: "#faq" },
];

const stats = [
  { value: "42%", label: "Downtime reduction" },
  { value: "3.1x", label: "Faster incident response" },
  { value: "$18M", label: "Avg. annual savings" },
  { value: "99.99%", label: "Platform uptime" },
];

const features = [
  {
    icon: Brain,
    title: "Elena",
    desc: "Ask questions in natural language, generate root-cause analyses, and let Elena draft work orders for your team.",
  },
  {
    icon: Gauge,
    title: "Predictive Maintenance",
    desc: "Anomaly detection on 50k+ sensors with remaining-useful-life estimates and confidence intervals.",
  },
  {
    icon: Layers,
    title: "Digital Twin",
    desc: "Live 3D-inspired visualisation of every line, cell, and asset — synced to the shop floor in milliseconds.",
  },
  {
    icon: LineChart,
    title: "Realtime Analytics",
    desc: "OEE, throughput, quality and energy dashboards designed for control-room-grade decision making.",
  },
  {
    icon: Workflow,
    title: "Workflow Automation",
    desc: "Trigger CMMS work orders, notify shifts, and dispatch technicians the moment a signal breaches threshold.",
  },
  {
    icon: ShieldCheck,
    title: "Enterprise Security",
    desc: "SOC 2 Type II, ISO 27001, on-prem or private cloud deployment, granular RBAC and audit trails.",
  },
];

const logos = ["SIEMENS", "BOSCH", "ABB", "HONEYWELL", "SCHNEIDER", "MITSUBISHI"];

const steps = [
  {
    n: "01",
    title: "Connect",
    desc: "Plug into PLCs, SCADA, historians and MES via 200+ pre-built connectors in minutes.",
    icon: Boxes,
  },
  {
    n: "02",
    title: "Model",
    desc: "Auto-generate a semantic asset model of your plant, from cell up to the enterprise.",
    icon: Cpu,
  },
  {
    n: "03",
    title: "Predict",
    desc: "Deploy pre-trained industrial models or fine-tune on your own operational history.",
    icon: Sparkles,
  },
  {
    n: "04",
    title: "Act",
    desc: "Automate escalation, dispatch, and reporting with your existing CMMS and ITSM tools.",
    icon: Zap,
  },
];

const testimonials = [
  {
    quote:
      "Iron Sight became the single pane of glass for our 14 plants. We cut unplanned downtime by 38% in the first six months.",
    name: "Sofia Marchetti",
    role: "VP Operations, NovaSteel Group",
  },
  {
    quote:
      "Elena writes better shift reports than our engineers. It's the closest thing to hiring 200 reliability specialists overnight.",
    name: "Kenji Watanabe",
    role: "Chief Reliability Officer, Kaizen Motors",
  },
  {
    quote:
      "Deployment took less than a quarter. The digital twin gave our C-suite realtime plant visibility for the first time ever.",
    name: "Marcus Reid",
    role: "CIO, Meridian Chemicals",
  },
];

const faqs = [
  {
    q: "How long does a typical deployment take?",
    a: "Most customers are live in a single plant within 3–5 weeks. Multi-site rollouts typically take one quarter per region using our reference architecture.",
  },
  {
    q: "Which industrial systems do you integrate with?",
    a: "We ship 200+ connectors including Siemens S7, Rockwell, OPC UA, MQTT, PI, Wonderware, SAP PM, Maximo, ServiceNow, and any REST/GraphQL endpoint.",
  },
  {
    q: "Can we self-host?",
    a: "Yes. Iron Sight runs on Kubernetes and supports on-prem, private cloud, and hybrid deployments with full offline inference.",
  },
  {
    q: "How is our data protected?",
    a: "SOC 2 Type II and ISO 27001 certified. All data is encrypted in transit and at rest, with per-tenant KMS keys and full audit trails.",
  },
];

function Landing() {
  return (
    <div className="relative min-h-dvh overflow-hidden bg-background text-foreground">
      {/* Nav */}
      <header className="sticky top-0 z-40">
        <div className="mx-auto mt-4 flex max-w-7xl items-center justify-between rounded-2xl glass px-4 py-2.5 md:mx-6">
          <div className="flex items-center gap-8">
            <BrandLogo />
            <nav className="hidden items-center gap-1 md:flex">
              {nav.map((n) => (
                <a
                  key={n.label}
                  href={n.href}
                  className="rounded-md px-3 py-1.5 text-sm text-muted-foreground transition hover:bg-white/5 hover:text-foreground"
                >
                  {n.label}
                </a>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-2">
            <Link
              to="/login"
              className="hidden rounded-md px-3 py-1.5 text-sm text-muted-foreground transition hover:text-foreground sm:inline-flex"
            >
              Sign in
            </Link>
            <Link to="/app">
              <Button size="sm" className="rounded-full shadow-glow">
                Launch console
                <ArrowRight className="ml-1 size-3.5" />
              </Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="relative mx-auto max-w-7xl px-6 pb-24 pt-20 md:pt-32">
        <div className="mx-auto max-w-4xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Badge
              variant="outline"
              className="mb-6 gap-2 rounded-full border-white/10 bg-white/5 px-3 py-1 text-xs backdrop-blur"
            >
              <span className="relative flex size-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex size-1.5 rounded-full bg-emerald-400" />
              </span>
              Now GA — Elena v3, with autonomous work-order drafting
            </Badge>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.05 }}
            className="text-balance text-5xl font-semibold leading-[1.05] tracking-tight md:text-7xl"
          >
            The operating system for{" "}
            <span className="text-gradient">modern factories.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.15 }}
            className="mx-auto mt-6 max-w-2xl text-pretty text-base leading-relaxed text-muted-foreground md:text-lg"
          >
            Iron Sight unifies predictive maintenance, digital twins, and Elena, a domain-tuned AI
            copilot, into one command center — engineered for the reliability of the plant floor and
            the scale of the enterprise.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.25 }}
            className="mt-9 flex flex-wrap items-center justify-center gap-3"
          >
            <Link to="/app">
              <Button size="lg" className="h-12 rounded-full px-6 shadow-glow">
                Open live demo
                <ArrowRight className="ml-2 size-4" />
              </Button>
            </Link>
            <Button
              size="lg"
              variant="outline"
              className="h-12 rounded-full border-white/15 bg-white/5 px-6 backdrop-blur hover:bg-white/10"
            >
              <Play className="mr-2 size-4" />
              Watch 90-sec tour
            </Button>
          </motion.div>
        </div>

        {/* Product preview */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.35 }}
          className="relative mx-auto mt-20 max-w-6xl"
        >
          <div className="absolute -inset-6 -z-10 rounded-[2rem] bg-gradient-to-br from-primary/25 via-accent/20 to-transparent blur-2xl" />
          <div className="glass-strong overflow-hidden rounded-2xl border border-white/10 shadow-elevated">
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-2.5">
              <div className="flex items-center gap-1.5">
                <div className="size-2.5 rounded-full bg-red-400/70" />
                <div className="size-2.5 rounded-full bg-yellow-400/70" />
                <div className="size-2.5 rounded-full bg-green-400/70" />
              </div>
              <div className="hidden items-center gap-2 rounded-md bg-white/5 px-3 py-1 text-xs text-muted-foreground md:flex">
                <Lock className="size-3" />
                console.ironsight.com/plant/nordic-01
              </div>
              <div className="text-xs text-muted-foreground">v3.4.0</div>
            </div>
            <div className="grid grid-cols-12 gap-0">
              <div className="col-span-3 hidden border-r border-white/10 p-4 md:block">
                <div className="mb-4 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
                  Plants
                </div>
                {["Nordic 01", "Osaka 04", "Rotterdam", "Detroit MFG", "Chennai 02"].map((p, i) => (
                  <div
                    key={p}
                    className={`mb-1 flex items-center justify-between rounded-md px-2 py-1.5 text-xs ${
                      i === 0 ? "bg-white/10 text-foreground" : "text-muted-foreground"
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      <Factory className="size-3.5" /> {p}
                    </span>
                    <span
                      className={`size-1.5 rounded-full ${
                        i === 2 ? "bg-yellow-400" : i === 3 ? "bg-red-400" : "bg-emerald-400"
                      }`}
                    />
                  </div>
                ))}
              </div>
              <div className="col-span-12 p-5 md:col-span-9">
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <div className="text-xs text-muted-foreground">Nordic 01 · Line A</div>
                    <div className="text-lg font-semibold tracking-tight">Realtime Overview</div>
                  </div>
                  <Badge className="bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/20">
                    Nominal
                  </Badge>
                </div>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  {[
                    { l: "OEE", v: "87.4%", t: "+2.1%" },
                    { l: "Throughput", v: "1,284/hr", t: "+3.8%" },
                    { l: "Energy", v: "412 kWh", t: "-6.4%" },
                    { l: "Alerts", v: "3", t: "2 open" },
                  ].map((k) => (
                    <div key={k.l} className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
                      <div className="text-[11px] text-muted-foreground">{k.l}</div>
                      <div className="mt-1 text-xl font-semibold tracking-tight">{k.v}</div>
                      <div className="text-[10px] text-emerald-400">{k.t}</div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 h-40 rounded-lg border border-white/10 bg-gradient-to-b from-white/[0.04] to-transparent p-4">
                  <MiniChart />
                </div>
              </div>
            </div>
          </div>

          {/* Floating cards */}
          <motion.div
            initial={{ opacity: 0, x: -20, y: 20 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            transition={{ delay: 0.9, duration: 0.6 }}
            className="absolute -left-4 top-1/3 hidden w-56 rounded-xl glass-strong p-3 shadow-elevated md:block"
          >
            <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
              <Sparkles className="size-3.5 text-primary" /> AI Insight
            </div>
            <div className="text-sm">
              Bearing #B-1147 shows <span className="text-warning">7.2σ vibration</span>. Predicted
              failure in <b>14–19 days</b>.
            </div>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: 20, y: 20 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            transition={{ delay: 1.1, duration: 0.6 }}
            className="absolute -right-4 bottom-8 hidden w-56 rounded-xl glass-strong p-3 shadow-elevated md:block"
          >
            <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
              <Activity className="size-3.5 text-emerald-400" /> Automation
            </div>
            <div className="text-sm">
              Auto-dispatched work order <b>WO-4821</b> to Reliability Team A.
            </div>
          </motion.div>
        </motion.div>
      </section>

      {/* Trusted by */}
      <section id="customers" className="relative border-y border-white/5 bg-white/[0.02] py-10">
        <div className="mx-auto max-w-7xl px-6">
          <div className="mb-6 text-center text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
            Trusted by the world's leading industrial teams
          </div>
          <div className="flex flex-wrap items-center justify-center gap-x-10 gap-y-4 opacity-70">
            {logos.map((l) => (
              <div
                key={l}
                className="text-sm font-semibold tracking-[0.25em] text-muted-foreground/80"
              >
                {l}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="relative mx-auto max-w-7xl px-6 py-24">
        <div className="grid gap-4 md:grid-cols-4">
          {stats.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.05 }}
              className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 backdrop-blur"
            >
              <div className="text-4xl font-semibold tracking-tight text-gradient">{s.value}</div>
              <div className="mt-2 text-sm text-muted-foreground">{s.label}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section id="features" className="relative mx-auto max-w-7xl px-6 pb-24">
        <div className="mx-auto max-w-2xl text-center">
          <div className="mb-3 text-xs font-medium uppercase tracking-[0.2em] text-primary">
            Platform
          </div>
          <h2 className="text-balance text-4xl font-semibold tracking-tight md:text-5xl">
            Everything you need to run a plant.{" "}
            <span className="text-muted-foreground">Nothing you don't.</span>
          </h2>
        </div>

        <div className="mt-14 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ delay: i * 0.04 }}
              className="group relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-b from-white/[0.04] to-transparent p-6 transition hover:border-primary/40"
            >
              <div className="absolute -right-16 -top-16 h-40 w-40 rounded-full bg-primary/10 blur-3xl transition group-hover:bg-primary/25" />
              <div className="relative">
                <div className="mb-4 grid size-10 place-items-center rounded-lg bg-white/5 ring-1 ring-white/10">
                  <f.icon className="size-5 text-primary" />
                </div>
                <div className="text-lg font-semibold tracking-tight">{f.title}</div>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{f.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="relative mx-auto max-w-7xl px-6 pb-24">
        <div className="mx-auto max-w-2xl text-center">
          <div className="mb-3 text-xs font-medium uppercase tracking-[0.2em] text-primary">
            How it works
          </div>
          <h2 className="text-balance text-4xl font-semibold tracking-tight md:text-5xl">
            From signal to action in seconds.
          </h2>
        </div>
        <div className="mt-14 grid gap-4 md:grid-cols-4">
          {steps.map((s) => (
            <div
              key={s.n}
              className="relative rounded-2xl border border-white/10 bg-white/[0.03] p-6"
            >
              <div className="mb-6 flex items-center justify-between">
                <span className="font-mono text-xs text-muted-foreground">{s.n}</span>
                <s.icon className="size-5 text-primary" />
              </div>
              <div className="text-lg font-semibold tracking-tight">{s.title}</div>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Testimonials */}
      <section className="relative mx-auto max-w-7xl px-6 pb-24">
        <div className="grid gap-4 md:grid-cols-3">
          {testimonials.map((t) => (
            <div
              key={t.name}
              className="rounded-2xl border border-white/10 bg-gradient-to-b from-white/[0.05] to-transparent p-6"
            >
              <div className="text-sm leading-relaxed">"{t.quote}"</div>
              <div className="mt-6 flex items-center gap-3">
                <div className="grid size-9 place-items-center rounded-full bg-gradient-to-br from-primary to-accent text-xs font-semibold text-white">
                  {t.name
                    .split(" ")
                    .map((n) => n[0])
                    .join("")}
                </div>
                <div>
                  <div className="text-sm font-medium">{t.name}</div>
                  <div className="text-xs text-muted-foreground">{t.role}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* FAQ */}
      <section id="faq" className="relative mx-auto max-w-4xl px-6 pb-24">
        <div className="mb-10 text-center">
          <div className="mb-3 text-xs font-medium uppercase tracking-[0.2em] text-primary">FAQ</div>
          <h2 className="text-4xl font-semibold tracking-tight md:text-5xl">
            Answers, from the deployment team.
          </h2>
        </div>
        <div className="divide-y divide-white/10 rounded-2xl border border-white/10 bg-white/[0.03]">
          {faqs.map((f) => (
            <details key={f.q} className="group px-6 py-5">
              <summary className="flex cursor-pointer list-none items-center justify-between text-left">
                <span className="text-base font-medium">{f.q}</span>
                <ChevronRight className="size-4 text-muted-foreground transition group-open:rotate-90" />
              </summary>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{f.a}</p>
            </details>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="relative mx-auto max-w-7xl px-6 pb-24">
        <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-primary/20 via-accent/10 to-transparent p-10 md:p-16">
          <div className="pointer-events-none absolute inset-0 bg-grid opacity-30" />
          <div className="relative mx-auto max-w-2xl text-center">
            <h3 className="text-balance text-4xl font-semibold tracking-tight md:text-5xl">
              Give your plants Elena, an AI copilot.
            </h3>
            <p className="mt-4 text-muted-foreground">
              See a live production environment in under 5 minutes. No credit card, no sales call.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-3">
              <Link to="/app">
                <Button size="lg" className="h-12 rounded-full px-6 shadow-glow">
                  Open the console <ArrowRight className="ml-2 size-4" />
                </Button>
              </Link>
              <Button
                size="lg"
                variant="outline"
                className="h-12 rounded-full border-white/15 bg-white/5 px-6 hover:bg-white/10"
              >
                Talk to sales
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative border-t border-white/10 bg-white/[0.02]">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-6 py-8 md:flex-row">
          <div className="flex items-center gap-4">
            <BrandLogo />
            <span className="text-xs text-muted-foreground">© 2026 Iron Sight Inc.</span>
          </div>
          <div className="flex items-center gap-6 text-xs text-muted-foreground">
            <a href="#" className="hover:text-foreground">Security</a>
            <a href="#" className="hover:text-foreground">Privacy</a>
            <a href="#" className="hover:text-foreground">Status</a>
            <a href="#" className="hover:text-foreground">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

function MiniChart() {
  const data = [12, 18, 14, 22, 19, 25, 21, 28, 26, 32, 30, 36, 34, 40, 38, 44, 42, 48];
  const max = Math.max(...data);
  const points = data
    .map((v, i) => `${(i / (data.length - 1)) * 100},${100 - (v / max) * 100}`)
    .join(" ");
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-full w-full">
      <defs>
        <linearGradient id="g1" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="oklch(0.68 0.19 250)" stopOpacity="0.4" />
          <stop offset="1" stopColor="oklch(0.68 0.19 250)" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        points={points}
        fill="none"
        stroke="oklch(0.75 0.15 210)"
        strokeWidth="1.2"
        vectorEffect="non-scaling-stroke"
      />
      <polygon points={`0,100 ${points} 100,100`} fill="url(#g1)" />
    </svg>
  );
}

import { BarChart3 as _bc } from "lucide-react";
void _bc;
