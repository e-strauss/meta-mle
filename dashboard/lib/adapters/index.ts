import "server-only";
import type { Adapter } from "../types";
import { metaMle } from "./meta-mle";

// Register another harness by adding its adapter here; the first that
// detects a directory owns it.
export const ADAPTERS: Adapter[] = [metaMle];

export function adapterFor(dir: string): Adapter | undefined {
  return ADAPTERS.find((a) => a.detect(dir));
}
