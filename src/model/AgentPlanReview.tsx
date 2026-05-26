import { AlertTriangle, Blocks, Cable, Check } from 'lucide-react'
import type { AgentGraphPlan, AgentGraphPreview } from '../core/agentic-graph'
import type { LaboLanguage } from '../studio/application-language'
import { agentPlanReviewText } from './agent-plan-review-text'

interface AgentPlanReviewProps {
  language: LaboLanguage
  plan: AgentGraphPlan
  preview: AgentGraphPreview
  onApply(): void
  onDiscard(): void
  onEditCard?(nodeId: string): void
}

export function AgentPlanReview({ language, onApply, onDiscard, onEditCard, plan, preview }: AgentPlanReviewProps) {
  const copy = agentPlanReviewText[language]
  const canApply = preview.acceptedBlocks.length > 0
    || preview.acceptedCreatedBlocks.length > 0
    || preview.accepted.length > 0
    || (plan.updatedBlocks?.length ?? 0) > 0
    || (plan.replacedBlocks?.length ?? 0) > 0
    || (plan.deletedBlocks?.length ?? 0) > 0
    || (plan.movedBlocks?.length ?? 0) > 0
    || preview.acceptedActions.some((action) => action.type !== 'layout')

  return <section aria-label={copy.title} className="agent-plan-review">
    <div className="ask-labo-result agent-plan-review-content">
      <section><h3>{copy.plan}</h3><p>{plan.summary}</p></section>

      {(plan.toolTrace?.length ?? 0) > 0 && <section>
        <h3>{copy.tools}</h3>
        <ul className="ask-labo-tool-trace">{plan.toolTrace!.map((item, index) => <li data-status={item.status} key={`${item.tool}-${index}`}><code>{item.tool}</code><span>{item.summary}</span></li>)}</ul>
      </section>}

      {preview.acceptedBlocks.length > 0 && <section>
        <h3>{preview.acceptedBlocks.length} {copy.atomic}{preview.acceptedBlocks.length === 1 ? '' : 's'} {language === 'en' ? 'ready' : 'prêts'}</h3>
        <div className="ask-labo-added-blocks">{preview.acceptedBlocks.map((block) => {
          const node = preview.graph.nodes.find((candidate) => candidate.id === block.nodeId)
          return <div key={block.nodeId}><Blocks size={13} /><span><strong>{node?.label ?? block.nodeId}</strong><code>{block.atomId}</code><small>{block.reason}</small></span>{onEditCard && <button aria-label={`Edit ${node?.label ?? block.nodeId}`} onClick={() => onEditCard(block.nodeId)} type="button">Edit</button>}</div>
        })}</div>
      </section>}

      {preview.acceptedCreatedBlocks.length > 0 && <section>
        <h3>{preview.acceptedCreatedBlocks.length} {copy.generated}{preview.acceptedCreatedBlocks.length === 1 ? '' : 's'} {language === 'en' ? 'ready' : 'prêtes'}</h3>
        <div className="ask-labo-created-blocks">{preview.acceptedCreatedBlocks.map((block) => {
          const node = preview.graph.nodes.find((candidate) => candidate.id === block.nodeId)
          return <div key={block.nodeId}><Blocks size={13} /><span><strong>{node?.label ?? block.label}</strong><code>{node?.code ?? block.pytorchModule}</code><small>{block.reason}</small></span>{onEditCard && <button aria-label={`Edit ${node?.label ?? block.label}`} onClick={() => onEditCard(block.nodeId)} type="button">Edit</button>}</div>
        })}</div>
      </section>}

      {preview.accepted.length > 0 && <section>
        <h3>{preview.accepted.length} {copy.elastic}{preview.accepted.length === 1 ? '' : 's'} {language === 'en' ? 'ready' : 'prêts'}</h3>
        <ul className="ask-labo-connections">{preview.accepted.map((connection) => <li key={`${connection.sourceId}.${connection.sourcePortId}-${connection.targetId}.${connection.targetPortId}`}><Cable size={13} /><div><code>{connection.sourceId}.{connection.sourcePortId}</code><span>→</span><code>{connection.targetId}.{connection.targetPortId}</code><small>{connection.reason}</small></div></li>)}</ul>
      </section>}

      {((plan.updatedBlocks?.length ?? 0) > 0 || (plan.replacedBlocks?.length ?? 0) > 0 || (plan.deletedBlocks?.length ?? 0) > 0 || (plan.movedBlocks?.length ?? 0) > 0) && <section>
        <h3>{copy.existing}</h3>
        {(plan.updatedBlocks ?? []).map((change) => <p key={`edit-${change.nodeId}`}><code>edit {change.nodeId}</code> · {change.reason}</p>)}
        {(plan.replacedBlocks ?? []).map((change) => <p key={`replace-${change.nodeId}`}><code>replace {change.nodeId} → {change.atomId}</code> · {change.reason}</p>)}
        {(plan.deletedBlocks?.length ?? 0) > 3 ? <p><code>{copy.deleteArchitecture}</code> · {plan.deletedBlocks!.length} {copy.cards} and their elastics</p> : (plan.deletedBlocks ?? []).map((change) => <p key={`delete-${change.nodeId}`}><code>delete {change.nodeId}</code> · {change.reason}</p>)}
        {(plan.movedBlocks ?? []).map((change) => <p key={`move-${change.nodeId}`}><code>move {change.nodeId}</code> · {change.reason}</p>)}
      </section>}

      {preview.acceptedActions.length > 0 && <section><h3>{copy.actions}</h3>{preview.acceptedActions.map((action, index) => <p key={`${action.type}-${index}`}><code>{action.type}{action.type === 'run' || action.type === 'run-selection' ? `:${action.mode}` : action.type === 'export' ? `:${action.kind}` : action.type === 'save-preset' ? `:${action.name}` : `:${action.scope}`}</code> · {action.reason}</p>)}</section>}

      {plan.missingBlocks.length > 0 && <section className="ask-labo-missing"><h3>{copy.missing}</h3>{plan.missingBlocks.map((block, index) => <div key={`${block.atomId ?? block.label}-${index}`}><AlertTriangle size={13} /><span><strong>{block.label}</strong><small>{block.reason}</small></span></div>)}</section>}

      {(preview.rejectedBlocks.length > 0 || preview.rejected.length > 0 || preview.rejectedMutations.length > 0 || plan.warnings.length > 0) && <section className="ask-labo-warnings">
        <h3>{copy.rejected}</h3>
        {preview.rejectedBlocks.map(({ block, reason }) => <p key={`${block.nodeId}-${reason}`}>{block.nodeId}: {reason}</p>)}
        {preview.rejected.map(({ connection, reason }) => <p key={`${connection.sourceId}-${connection.targetId}-${reason}`}>{connection.sourceId} → {connection.targetId}: {reason}</p>)}
        {preview.rejectedMutations.map(({ nodeId, action, reason }, index) => <p key={`${nodeId ?? action?.type ?? 'mutation'}-${index}`}>{nodeId ?? action?.type}: {reason}</p>)}
        {plan.warnings.map((warning) => <p key={warning}>{warning}</p>)}
      </section>}
    </div>

    <footer className="ask-labo-actions agent-plan-review-actions">
      <span className="ask-labo-approval-summary">{copy.ready} · {preview.acceptedBlocks.length + preview.acceptedCreatedBlocks.length} {copy.cards} · {preview.accepted.length} {copy.elastic}{preview.accepted.length === 1 ? '' : 's'}</span>
      <button className="ask-labo-cancel" onClick={onDiscard} type="button">{copy.discard}</button>
      <button aria-label="Apply full graph plan" className="ask-labo-apply" disabled={!canApply} onClick={onApply} type="button"><Check size={13} />{copy.apply}</button>
    </footer>
  </section>
}
