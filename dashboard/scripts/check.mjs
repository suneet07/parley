import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");

let errorCount = 0;

function reportError(file, line, msg) {
  const relPath = path.relative(rootDir, file);
  console.error(`Violation in ${relPath}${line ? `:${line}` : ""}: ${msg}`);
  errorCount++;
}

// 1. Parse globals.css for defined class names (.name selectors)
const globalsCssPath = path.join(rootDir, "app/globals.css");
if (!fs.existsSync(globalsCssPath)) {
  console.error("globals.css not found at", globalsCssPath);
  process.exit(1);
}
const globalsCss = fs.readFileSync(globalsCssPath, "utf-8");

const globalsClasses = new Set();
const classSelectorRegex = /\.([a-zA-Z_-][a-zA-Z0-9_-]*)/g;
for (const match of globalsCss.matchAll(classSelectorRegex)) {
  globalsClasses.add(match[1]);
}

const EXACT_TOKENS = new Set([
  "flex",
  "inline-flex",
  "grid",
  "hidden",
  "block",
  "inline",
  "text-xs",
  "text-sm",
  "text-base",
  "text-lg",
  "text-xl",
  "text-2xl",
  "truncate",
  "text-left",
  "text-right",
  "text-center",
  "rounded",
  "border",
  "border-t",
  "border-b",
  "sticky",
  "relative",
  "absolute",
  "sr-only",
  "underline",
  "no-underline",
  "uppercase",
  "capitalize",
  "italic",
  "transition",
  // Theme color utility tokens
  "bg-surface",
  "bg-surface-2",
  "text-muted",
  "text-faint",
  "text-ink",
  "text-signal",
  "text-ok",
  "text-warn",
  "text-danger",
  "border-line",
  "border-line-strong",
]);

const ALLOWED_PREFIXES = [
  "items-",
  "justify-",
  "self-",
  "content-",
  "gap-",
  "p-",
  "px-",
  "py-",
  "pt-",
  "pb-",
  "pl-",
  "pr-",
  "m-",
  "mx-",
  "my-",
  "mt-",
  "mb-",
  "ml-",
  "mr-",
  "w-",
  "h-",
  "min-w-",
  "max-w-",
  "min-h-",
  "size-",
  "col-span-",
  "row-span-",
  "grid-cols-",
  "grid-rows-",
  "flex-",
  "shrink-",
  "grow-",
  "basis-",
  "order-",
  "font-",
  "leading-",
  "tracking-",
  "whitespace-",
  "break-",
  "overflow-",
  "divide-",
  "top-",
  "inset-",
  "z-",
  "cursor-",
  "select-",
  "list-",
  "space-y-",
  "space-x-",
  "align-",
  "opacity-",
];

const MODIFIER_PREFIXES = ["lg:", "md:", "sm:", "xl:", "hover:", "focus:"];

function isValidClassToken(token) {
  if (!token) return true;
  let t = token.trim();
  if (!t) return true;

  // Strip modifier prefixes
  let stripped = true;
  while (stripped) {
    stripped = false;
    for (const mod of MODIFIER_PREFIXES) {
      if (t.startsWith(mod)) {
        t = t.slice(mod.length);
        stripped = true;
        break;
      }
    }
  }

  if (globalsClasses.has(t)) return true;
  if (EXACT_TOKENS.has(t)) return true;
  for (const prefix of ALLOWED_PREFIXES) {
    if (t.startsWith(prefix)) return true;
  }
  return false;
}

function findTsxFiles(dir) {
  const results = [];
  if (!fs.existsSync(dir)) return results;
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...findTsxFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith(".tsx")) {
      results.push(fullPath);
    }
  }
  return results;
}

const appFiles = findTsxFiles(path.join(rootDir, "app"));
const componentFiles = findTsxFiles(path.join(rootDir, "components"));
const allTsxFiles = [...appFiles, ...componentFiles];

// Regular expressions for rules 2, 3, 4
const hexColorRegex = /#([0-9a-fA-F]{3,8})\b/;
const stockColorRegex =
  /\b(text|bg|border)-(gray|slate|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d/;
const booleanAttrRegex =
  /\b(data-active|data-interrupted)\s*=\s*(\{([\s\S]*?)\}|"[^"]*"|'[^']*')/g;

for (const filePath of allTsxFiles) {
  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.split("\n");

  // Check rule 2 & rule 3 line-by-line
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineNum = i + 1;

    // Rule 2: No raw hex colors
    const hexMatch = line.match(hexColorRegex);
    if (hexMatch) {
      reportError(
        filePath,
        lineNum,
        `Raw hex color "${hexMatch[0]}" is not allowed. Use theme tokens.`
      );
    }

    // Rule 2: No stock palette colors
    const stockMatch = line.match(stockColorRegex);
    if (stockMatch) {
      reportError(
        filePath,
        lineNum,
        `Stock palette color "${stockMatch[0]}" is not allowed. Use theme tokens.`
      );
    }

    // Rule 3: style={{ only on lines containing lat-seg or hbar-fill (or the line right after)
    if (line.includes("style={{")) {
      const prevLine = i > 0 ? lines[i - 1] : "";
      const allowed =
        line.includes("lat-seg") ||
        line.includes("hbar-fill") ||
        prevLine.includes("lat-seg") ||
        prevLine.includes("hbar-fill");
      if (!allowed) {
        reportError(
          filePath,
          lineNum,
          `style={{ is only allowed on lines containing lat-seg or hbar-fill.`
        );
      }
    }
  }

  // Rule 4: Every data-active=/aria-pressed=/data-interrupted= value is wrapped in String( or is string literal
  for (const match of content.matchAll(booleanAttrRegex)) {
    const attrName = match[1];
    const rawVal = match[2];
    if (rawVal.startsWith("{")) {
      const expr = match[3].trim();
      if (!expr.startsWith("String(")) {
        reportError(
          filePath,
          null,
          `${attrName} value must be wrapped in String(...), got: ${rawVal}`
        );
      }
    }
  }

  // Rule 1: Validate all className tokens
  const classNameAttrRegex = /className\s*=\s*("([^"]*)"|'([^']*)'|\{([\s\S]*?)\})/g;
  for (const match of content.matchAll(classNameAttrRegex)) {
    let rawStrings = [];
    if (match[2] !== undefined) {
      rawStrings.push(match[2]);
    } else if (match[3] !== undefined) {
      rawStrings.push(match[3]);
    } else if (match[4] !== undefined) {
      const expr = match[4];
      const strLiteralRegex = /"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|`((?:[^`\\]|\\.)*)`/g;
      for (const strMatch of expr.matchAll(strLiteralRegex)) {
        if (strMatch[1] !== undefined) {
          rawStrings.push(strMatch[1]);
        } else if (strMatch[2] !== undefined) {
          rawStrings.push(strMatch[2]);
        } else if (strMatch[3] !== undefined) {
          const parts = strMatch[3].split(/\$\{[^}]*\}/);
          rawStrings.push(...parts);
        }
      }
    }

    for (const rawStr of rawStrings) {
      const tokens = rawStr.split(/\s+/).filter(Boolean);
      for (const token of tokens) {
        if (!isValidClassToken(token)) {
          reportError(
            filePath,
            null,
            `Unknown className token "${token}". Must be in globals.css, allowed Tailwind utilities, or theme color tokens.`
          );
        }
      }
    }
  }
}

if (errorCount > 0) {
  console.error(`\nCheck failed with ${errorCount} violation(s).`);
  process.exit(1);
} else {
  console.log(`Check passed cleanly on ${allTsxFiles.length} tsx files.`);
  process.exit(0);
}
