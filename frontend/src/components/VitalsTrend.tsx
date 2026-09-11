import { useMemo } from "react";
import type { VitalsRecord } from "@/lib/api";
import { formatDateTime } from "@/lib/health";

/**
 * Blood pressure across recent readings.
 *
 * Hand-drawn SVG rather than a charting library: two series over at most
 * twenty points does not justify pulling ~300 KB of recharts back into the
 * bundle.
 */

const WIDTH = 520;
const HEIGHT = 170;
const PADDING = { top: 14, right: 12, bottom: 26, left: 34 };

// Reference lines for the thresholds that matter clinically.
const THRESHOLDS = [
  { value: 140, label: "140", series: "systolic" },
  { value: 90, label: "90", series: "diastolic" },
];

export function VitalsTrend({ records }: { records: VitalsRecord[] }) {
  // The API returns newest first; a time axis needs oldest first.
  const points = useMemo(() => [...records].reverse(), [records]);

  if (points.length === 0) {
    return (
      <p className="py-10 text-center text-sm text-muted-foreground">
        No readings recorded yet.
      </p>
    );
  }

  if (points.length === 1) {
    const only = points[0];
    return (
      <div className="py-8 text-center">
        <p className="tabular text-3xl font-bold">
          {only.systolic_bp}/{only.diastolic_bp}
          <span className="ml-1 text-sm font-normal text-muted-foreground">mmHg</span>
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          Record another reading to see how it changes over time.
        </p>
      </div>
    );
  }

  const values = points.flatMap((record) => [record.systolic_bp, record.diastolic_bp]);
  const min = Math.min(...values, 60) - 5;
  const max = Math.max(...values, 150) + 5;

  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;

  const x = (index: number) =>
    PADDING.left + (index / (points.length - 1)) * plotWidth;
  const y = (value: number) =>
    PADDING.top + plotHeight - ((value - min) / (max - min)) * plotHeight;

  const line = (pick: (record: VitalsRecord) => number) =>
    points.map((record, index) => `${index === 0 ? "M" : "L"}${x(index)},${y(pick(record))}`).join(" ");

  const axisLabels = [max, (max + min) / 2, min].map(Math.round);

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full"
        role="img"
        aria-label={`Blood pressure across your last ${points.length} readings, from ${formatDateTime(points[0].created_at)} to ${formatDateTime(points[points.length - 1].created_at)}`}
      >
        {/* Horizontal grid and axis labels */}
        {axisLabels.map((value) => (
          <g key={value}>
            <line
              x1={PADDING.left}
              x2={WIDTH - PADDING.right}
              y1={y(value)}
              y2={y(value)}
              className="stroke-border"
              strokeWidth={1}
            />
            <text
              x={PADDING.left - 6}
              y={y(value) + 3}
              textAnchor="end"
              className="fill-muted-foreground text-[9px]"
            >
              {value}
            </text>
          </g>
        ))}

        {/* Clinical thresholds */}
        {THRESHOLDS.filter((threshold) => threshold.value > min && threshold.value < max).map(
          (threshold) => (
            <line
              key={threshold.series}
              x1={PADDING.left}
              x2={WIDTH - PADDING.right}
              y1={y(threshold.value)}
              y2={y(threshold.value)}
              className="stroke-warning/50"
              strokeWidth={1}
              strokeDasharray="4 3"
            />
          ),
        )}

        {/* Series */}
        <path
          d={line((record) => record.systolic_bp)}
          fill="none"
          className="stroke-primary"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d={line((record) => record.diastolic_bp)}
          fill="none"
          className="stroke-purple-500"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {points.map((record, index) => (
          <g key={record.id}>
            <circle cx={x(index)} cy={y(record.systolic_bp)} r={3} className="fill-primary" />
            <circle cx={x(index)} cy={y(record.diastolic_bp)} r={3} className="fill-purple-500" />
            <title>
              {`${formatDateTime(record.created_at)}: ${record.systolic_bp}/${record.diastolic_bp} mmHg`}
            </title>
          </g>
        ))}

        {/* Only the endpoints are labelled; intermediate dates would collide. */}
        <text
          x={PADDING.left}
          y={HEIGHT - 8}
          className="fill-muted-foreground text-[9px]"
        >
          oldest
        </text>
        <text
          x={WIDTH - PADDING.right}
          y={HEIGHT - 8}
          textAnchor="end"
          className="fill-muted-foreground text-[9px]"
        >
          latest
        </text>
      </svg>

      <figcaption className="mt-3 flex items-center justify-center gap-5 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded bg-primary" />
          Systolic
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded bg-purple-500" />
          Diastolic
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0 w-4 border-t border-dashed border-warning" />
          Thresholds
        </span>
      </figcaption>
    </figure>
  );
}
