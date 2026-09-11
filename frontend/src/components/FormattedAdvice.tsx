import { Fragment, useMemo } from "react";
import { cn } from "@/lib/utils";

/**
 * Renders the light Markdown the model returns.
 *
 * The previous implementation split the whole string on "**" and bolded every
 * odd segment, so a single stray asterisk inverted the formatting for the rest
 * of the message, and lists and headings rendered as raw text. This handles
 * headings, bullets, numbered steps, and inline bold and code.
 *
 * Handles headings, bullets, numbered steps, horizontal rules, tables, and
 * inline bold and code. Tables matter in particular: the model reaches for one
 * whenever it compares a reading against a normal range, and they previously
 * rendered as raw pipe characters.
 *
 * Only these constructs are interpreted and everything else is rendered as
 * plain text — no HTML from the model ever reaches the DOM.
 */

type Block =
  | { kind: "heading"; text: string }
  | { kind: "paragraph"; text: string }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "table"; head: string[]; rows: string[][] }
  | { kind: "rule" };

/** Split a Markdown table row into cells, dropping the outer pipes. */
function splitRow(line: string): string[] {
  return line
    .replace(/^\s*\|/, "")
    .replace(/\|\s*$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

/** A `|---|:---:|` separator marks the line above as a header row. */
function isTableDivider(line: string): boolean {
  return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line) && line.includes("-");
}

function isTableRow(line: string): boolean {
  return line.trim().startsWith("|") && line.trim().endsWith("|");
}

/** A few HTML entities the model occasionally emits instead of characters. */
const ENTITIES: Record<string, string> = {
  "&nbsp;": " ",
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&#39;": "'",
  "&apos;": "'",
};

/**
 * Normalise the stray HTML the model mixes into its Markdown.
 *
 * It is asked for Markdown only, but still reaches for `<br>` — often packing
 * a whole bullet list onto one line as `• A <br>• B <br>• C`. Left alone that
 * renders the tag as visible text *and* collapses the list into a single item,
 * because the parser only ever sees one line.
 *
 * Converting `<br>` to a real newline before parsing fixes both at once. Bold
 * tags become their Markdown equivalent so the emphasis survives; the few
 * other inline tags are dropped rather than shown to the reader.
 */
function normaliseHtml(source: string): string {
  let text = source.replace(/\r\n/g, "\n");

  text = text.replace(/<br\s*\/?>/gi, "\n");
  text = text.replace(/<\/?(?:b|strong)>/gi, "**");
  text = text.replace(/<\/?(?:i|em|u|span|small|font)[^>]*>/gi, "");
  text = text.replace(/<\/?p>/gi, "\n");

  for (const [entity, character] of Object.entries(ENTITIES)) {
    text = text.split(entity).join(character);
  }
  return text;
}

function parseBlocks(source: string): Block[] {
  const blocks: Block[] = [];
  const lines = normaliseHtml(source).split("\n");

  let paragraph: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushParagraph = () => {
    if (paragraph.length > 0) {
      blocks.push({ kind: "paragraph", text: paragraph.join(" ").trim() });
      paragraph = [];
    }
  };

  const flushList = () => {
    if (list && list.items.length > 0) {
      blocks.push({ kind: "list", ordered: list.ordered, items: list.items });
    }
    list = null;
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (trimmed === "") {
      flushParagraph();
      flushList();
      continue;
    }

    // A horizontal rule separates sections in the model's longer answers.
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "rule" });
      continue;
    }

    // Tables: a row, a divider, then body rows until the block ends. The
    // model reaches for these often when comparing a reading to a normal
    // range, and they previously rendered as raw pipe characters.
    if (isTableRow(trimmed) && isTableDivider(lines[index + 1] ?? "")) {
      flushParagraph();
      flushList();

      const head = splitRow(trimmed);
      const rows: string[][] = [];
      index += 2; // step past the header and its divider

      while (index < lines.length && isTableRow(lines[index].trim())) {
        rows.push(splitRow(lines[index].trim()));
        index += 1;
      }
      index -= 1; // the outer loop advances again

      blocks.push({ kind: "table", head, rows });
      continue;
    }

    const heading = /^#{1,6}\s+(.*)$/.exec(trimmed);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "heading", text: heading[1] });
      continue;
    }

    // A line that is entirely bold also reads as a heading.
    const boldHeading = /^\*\*(.+)\*\*:?$/.exec(trimmed);
    if (boldHeading) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "heading", text: boldHeading[1] });
      continue;
    }

    const bullet = /^[-*•]\s+(.*)$/.exec(trimmed);
    if (bullet) {
      flushParagraph();
      if (!list || list.ordered) {
        flushList();
        list = { ordered: false, items: [] };
      }
      list.items.push(bullet[1]);
      continue;
    }

    const numbered = /^\d+[.)]\s+(.*)$/.exec(trimmed);
    if (numbered) {
      flushParagraph();
      if (!list || !list.ordered) {
        flushList();
        list = { ordered: true, items: [] };
      }
      list.items.push(numbered[1]);
      continue;
    }

    flushList();
    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();
  return blocks;
}

/** Apply inline **bold** and `code` to a single line of text. */
function renderInline(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);

  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={index} className="font-semibold text-foreground">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code key={index} className="rounded bg-muted px-1 py-0.5 text-[0.9em]">
          {part.slice(1, -1)}
        </code>
      );
    }
    return <Fragment key={index}>{part}</Fragment>;
  });
}

export function FormattedAdvice({ text, className }: { text: string; className?: string }) {
  const blocks = useMemo(() => parseBlocks(text), [text]);

  return (
    <div className={cn("space-y-3 text-[0.95rem] leading-relaxed", className)}>
      {blocks.map((block, index) => {
        if (block.kind === "heading") {
          return (
            <h4 key={index} className="text-base font-semibold text-foreground">
              {renderInline(block.text)}
            </h4>
          );
        }

        if (block.kind === "rule") {
          return <hr key={index} className="border-border" />;
        }

        if (block.kind === "table") {
          return (
            // Tables are the one thing here that can exceed the column width,
            // so this scrolls on its own rather than the whole page.
            <div key={index} className="-mx-1 overflow-x-auto px-1">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr>
                    {block.head.map((cell, cellIndex) => (
                      <th
                        key={cellIndex}
                        scope="col"
                        className="border-b-2 border-border px-3 py-2 text-left font-semibold"
                      >
                        {renderInline(cell)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {block.rows.map((row, rowIndex) => (
                    <tr key={rowIndex} className="border-b border-border last:border-0">
                      {row.map((cell, cellIndex) => (
                        <td key={cellIndex} className="px-3 py-2 align-top">
                          {renderInline(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }

        if (block.kind === "list") {
          const ListTag = block.ordered ? "ol" : "ul";
          return (
            <ListTag
              key={index}
              className={cn(
                "space-y-1.5 pl-5",
                block.ordered ? "list-decimal" : "list-disc",
                "marker:text-primary/70",
              )}
            >
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex} className="pl-1">
                  {renderInline(item)}
                </li>
              ))}
            </ListTag>
          );
        }

        return <p key={index}>{renderInline(block.text)}</p>;
      })}
    </div>
  );
}
