import { lineDiff } from "../lib/diff";

interface Props {
  before: string[];
  after: string[] | null;
}

export function PersonaDiff({ before, after }: Props) {
  if (after === null) {
    return (
      <div className="persona-diff">
        <p className="no-rewrite-note">
          No rewrite fired for this agent — the card shown below is unchanged across onset.
        </p>
        {before.map((line, i) => (
          <div className="diff-line unchanged" key={i}>
            <span className="mark"> </span>
            {line}
          </div>
        ))}
      </div>
    );
  }

  const ops = lineDiff(before, after);
  return (
    <div className="persona-diff">
      {ops.map((op, i) => (
        <div className={`diff-line ${op.type}`} key={i}>
          <span className="mark">{op.type === "removed" ? "−" : op.type === "added" ? "+" : " "}</span>
          {op.text}
        </div>
      ))}
    </div>
  );
}
