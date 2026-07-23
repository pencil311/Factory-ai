import { createFileRoute, Outlet } from "@tanstack/react-router";

import { AppSidebar } from "@/components/app-sidebar";

export const Route = createFileRoute("/app")({
  component: AppLayout,
});

function AppLayout() {
  return (
    <div className="flex min-h-dvh w-full bg-background">
      <AppSidebar />
      <div className="relative flex min-w-0 flex-1 flex-col">
        <Outlet />
      </div>
    </div>
  );
}
