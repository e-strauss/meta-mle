import Link from "next/link";
import { cookies } from "next/headers";
import { logout } from "@/app/actions/auth";
import ThemeSelect from "@/components/theme-select";
import { parseTheme, THEME_COOKIE } from "@/lib/theme";

export default async function Header() {
  const theme = parseTheme((await cookies()).get(THEME_COOKIE)?.value);
  return (
    <header className="top">
      <Link href="/" className="brand">ML search dashboard</Link>
      <Link href="/">Runs</Link>
      <Link href="/new">New run</Link>
      <span className="spacer" />
      <ThemeSelect initialTheme={theme} />
      <form action={logout}><button type="submit">Sign out</button></form>
    </header>
  );
}
