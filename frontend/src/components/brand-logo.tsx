import { Link } from "@tanstack/react-router";

export function BrandLogo({ compact = false }: { compact?: boolean }) {
  return (
    <Link to="/" className="group flex items-center gap-2.5">
      <div className="relative grid h-8 w-8 place-items-center rounded-lg bg-primary">
        <div className="h-3 w-3 rounded-sm bg-primary-foreground" />
      </div>
      {!compact && (
        <div className="flex flex-col leading-none">
          <span className="text-sm font-semibold tracking-tight">Iron Sight</span>
          <span className="text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
            Elena
          </span>
        </div>
      )}
    </Link>
  );
}
