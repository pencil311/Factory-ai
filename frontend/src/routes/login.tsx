import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ArrowRight, Github, Loader2, Mail } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { BrandLogo } from "@/components/brand-logo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign in — Iron Sight" },
      { name: "description", content: "Sign in to your Iron Sight enterprise workspace." },
      { property: "og:title", content: "Sign in — Iron Sight" },
      { property: "og:description", content: "Access your industrial AI command center." },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      toast.success("Welcome back", { description: "Loading your workspace…" });
      navigate({ to: "/app" });
    }, 900);
  };

  return (
    <div className="relative grid min-h-dvh grid-cols-1 lg:grid-cols-2">
      {/* Left: illustration */}
      <div className="relative hidden overflow-hidden bg-[oklch(0.13_0.03_265)] lg:block">
        <div className="relative z-10 flex h-full flex-col justify-between p-10">
          <BrandLogo />

          <div className="max-w-md">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-muted-foreground backdrop-blur">
              <span className="size-1.5 rounded-full bg-emerald-400" /> All systems operational
            </div>
            <h2 className="text-4xl font-semibold tracking-tight">
              The command center for <span className="text-gradient">industrial intelligence.</span>
            </h2>
            <p className="mt-4 text-muted-foreground">
              Predict, prevent, and act — across every plant, line and asset. Trusted by industrial
              leaders in 42 countries.
            </p>

            <div className="mt-10 space-y-3">
              {[
                { k: "OEE across plants", v: "87.4%" },
                { k: "Downtime avoided (30d)", v: "412 hrs" },
                { k: "Open work orders", v: "128" },
              ].map((r) => (
                <div
                  key={r.k}
                  className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.04] p-4 backdrop-blur"
                >
                  <div className="text-sm text-muted-foreground">{r.k}</div>
                  <div className="font-mono text-lg tracking-tight">{r.v}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="text-xs text-muted-foreground">
            © 2026 Iron Sight Inc. — SOC 2 · ISO 27001
          </div>
        </div>
      </div>

      {/* Right: form */}
      <div className="relative flex items-center justify-center bg-background px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-sm"
        >
          <div className="lg:hidden">
            <BrandLogo />
          </div>

          <div className="mt-8">
            <h1 className="text-3xl font-semibold tracking-tight">Sign in</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              Welcome back. Enter your details to access the console.
            </p>
          </div>

          <div className="mt-8 grid grid-cols-2 gap-3">
            <Button variant="outline" className="h-10 border-white/10 bg-white/[0.03]">
              <Github className="mr-2 size-4" /> GitHub
            </Button>
            <Button variant="outline" className="h-10 border-white/10 bg-white/[0.03]">
              <Mail className="mr-2 size-4" /> SSO
            </Button>
          </div>

          <div className="my-6 flex items-center gap-3">
            <Separator className="flex-1 bg-white/10" />
            <span className="text-[10px] uppercase tracking-widest text-muted-foreground">or</span>
            <Separator className="flex-1 bg-white/10" />
          </div>

          <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="email" className="text-xs">
                Work email
              </Label>
              <Input
                id="email"
                type="email"
                required
                defaultValue="sofia@novasteel.com"
                className="h-11 border-white/10 bg-white/[0.03]"
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label htmlFor="password" className="text-xs">
                  Password
                </Label>
                <a href="#" className="text-xs text-primary hover:underline">
                  Forgot?
                </a>
              </div>
              <Input
                id="password"
                type="password"
                required
                defaultValue="demo-password"
                className="h-11 border-white/10 bg-white/[0.03]"
              />
            </div>

            <div className="flex items-center gap-2">
              <Checkbox id="remember" defaultChecked />
              <Label htmlFor="remember" className="text-xs text-muted-foreground">
                Keep me signed in on this device
              </Label>
            </div>

            <Button type="submit" className="h-11 w-full rounded-md shadow-glow" disabled={loading}>
              {loading ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <>
                  Continue <ArrowRight className="ml-1 size-4" />
                </>
              )}
            </Button>
          </form>

          <p className="mt-6 text-center text-xs text-muted-foreground">
            Don't have an account?{" "}
            <a href="#" className="text-primary hover:underline">
              Request access
            </a>
          </p>

          <div className="mt-10 text-center">
            <Link to="/" className="text-xs text-muted-foreground hover:text-foreground">
              ← Back to site
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
