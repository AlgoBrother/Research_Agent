import type { ReactNode } from "react";

export const metadata = {
  title: "ArXAgent",
  description: "Multi-agent research assistant",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#fafafa" }}>{children}</body>
    </html>
  );
}
