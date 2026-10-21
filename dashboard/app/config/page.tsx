"use client";

import { Badge } from "@/components/Badge";
import { Empty, ErrorBox, Loading } from "@/components/States";
import { useConfig } from "@/lib/api";

export default function ConfigPage() {
  const { data: config, error, isLoading } = useConfig();

  if (isLoading) {
    return <Loading />;
  }

  if (error) {
    return <ErrorBox message={error.message || "Failed to load configuration."} />;
  }

  if (!config) {
    return <Empty>No configuration found.</Empty>;
  }

  const { business, tools, knowledge } = config;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Config</h1>
          <p className="page-sub">Business profile, policies, tools, and knowledge base</p>
        </div>
      </div>

      <div className="flex flex-col gap-6">
        <div className="card card-pad">
          <h2 className="text-sm font-semibold mb-3">Business</h2>
          <dl className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <dt className="dim text-xs font-medium mb-1">Name</dt>
              <dd className="font-medium">{business.business}</dd>
            </div>
            <div>
              <dt className="dim text-xs font-medium mb-1">Hours</dt>
              <dd>
                {business.hours?.open} – {business.hours?.close} ({business.hours?.tz})
              </dd>
            </div>
            <div className="md:col-span-2">
              <dt className="dim text-xs font-medium mb-1">Persona</dt>
              <dd className="dim leading-relaxed">{business.persona}</dd>
            </div>
            <div className="md:col-span-2">
              <dt className="dim text-xs font-medium mb-1">Escalation</dt>
              <dd>
                Transfer to:{" "}
                <strong className="mono">
                  {business.escalation?.transfer_to || "None"}
                </strong>
                {" · "}
                Triggers:{" "}
                <span className="dim">
                  {business.escalation?.on?.join(", ") || "None"}
                </span>
              </dd>
            </div>
          </dl>
        </div>

        <div className="card card-pad">
          <h2 className="text-sm font-semibold mb-3">Policies</h2>
          {business.policies && Object.keys(business.policies).length > 0 ? (
            <div className="overflow-x-auto">
              <table className="table">
                <thead>
                  <tr>
                    <th scope="col">Policy key</th>
                    <th scope="col">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(business.policies).map(([key, val]) => (
                    <tr key={key}>
                      <td className="mono">{key}</td>
                      <td>{String(val ?? "—")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="dim text-sm">No custom policies configured.</p>
          )}
        </div>

        <div className="card card-pad">
          <h2 className="text-sm font-semibold mb-3">Tools</h2>
          {tools && tools.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="table">
                <thead>
                  <tr>
                    <th scope="col">Name</th>
                    <th scope="col">Kind</th>
                    <th scope="col">Verification</th>
                    <th scope="col">Confirmation</th>
                    <th scope="col">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {tools.map((tool) => (
                    <tr key={tool.name}>
                      <td>
                        <span className="mono">{tool.name}</span>
                      </td>
                      <td>
                        <Badge>{tool.kind}</Badge>
                      </td>
                      <td>{tool.requires_verification ? "✓" : "—"}</td>
                      <td>{tool.requires_confirmation ? "✓" : "—"}</td>
                      <td>
                        <span className="dim">{tool.description}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="dim text-sm">No tools registered.</p>
          )}
        </div>

        <div className="card card-pad">
          <h2 className="text-sm font-semibold mb-3">Knowledge files</h2>
          {knowledge && knowledge.length > 0 ? (
            <ul className="list-disc pl-5 space-y-1">
              {knowledge.map((file) => (
                <li key={file} className="mono">
                  {file}
                </li>
              ))}
            </ul>
          ) : (
            <p className="dim text-sm">No knowledge files configured.</p>
          )}
        </div>
      </div>

      <p className="dim text-xs mt-6">
        Edit examples/.../business.yaml and restart to change config.
      </p>
    </div>
  );
}
