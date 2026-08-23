import type { BillingBlock } from '@hermes/shared'
import { registryBackendScopeKey } from '@hermes/shared'
import type { HermesSkin } from '@hermes/shared/skin'
import type { QueryClient } from '@tanstack/react-query'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { readActivePreview } from '@/app/chat/right-rail/preview-reader'
import { writeAgentTerminalChunk } from '@/app/right-sidebar/terminal/agent-terminal-stream'
import { readActiveTerminal } from '@/app/right-sidebar/terminal/buffer'
import { closeAgentTerminalByProc } from '@/app/right-sidebar/terminal/terminals'
import { burstVibeHearts } from '@/components/chat/vibe-hearts'
import { translateNow } from '@/i18n'
import { type GatewayEventPayload, textPart } from '@/lib/chat-messages'
import { coerceGatewayText, coerceThinkingText, normalizePersonalityValue } from '@/lib/chat-runtime'
import { playCompletionSound } from '@/lib/completion-sound'
import {
  approvalReplaySessionId,
  resolveGatewayEventSessionId,
  UNSCOPED_STREAM_EVENT_TYPES
} from '@/lib/gateway-events'
import { triggerHaptic } from '@/lib/haptics'
import { modelOptionsQueryKey } from '@/lib/model-options'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { type AgentNoticePayload, clearAgentNotice, nativeNoticeInput, showAgentNotice } from '@/store/agent-notices'
import { reconcileApprovalModeForProfile } from '@/store/approval-mode'
import { billingCtaLabel, clearBillingBlock, runBillingRecovery, setBillingBlock } from '@/store/billing-block'
import { clearClarifyRequest, normalizeChoices, setClarifyRequest, warnDroppedChoices } from '@/store/clarify'
import { setSessionCompacting } from '@/store/compaction'
import { refreshBackgroundProcesses } from '@/store/composer-status'
import { $gateway, activeGatewayConnectionId } from '@/store/gateway'
import { applyGoalStatusText } from '@/store/goals'
import {
  notifyCronChanged,
  notifyPairingChanged,
  notifyPetChanged,
  notifyPlatformsChanged,
  notifySessionsChanged,
  type PetChangeMeta,
  setChangeEventsAvailable
} from '@/store/live-sync'
import { setMcpSetupRequest } from '@/store/mcp-setup'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { isDiskFullErrorMessage, notify, notifyError } from '@/store/notifications'
import { requestDesktopOnboarding, requestDesktopOnboardingForCredentialWarning } from '@/store/onboarding'
import { revealDesktopPane } from '@/store/pane-focus'
import { flashPetActivity, markPetUnread, setPetActivity } from '@/store/pet'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { followActiveSessionCwd } from '@/store/projects'
import {
  clearAllPrompts,
  receiveApprovalRequest,
  replayPendingApproval,
  setSecretRequest,
  setSudoRequest
} from '@/store/prompts'
import { providerWaitText, setSessionProviderWait } from '@/store/provider-wait'
import { recordAgentReaction } from '@/store/reactions-local'
import {
  $currentCwd,
  $currentModel,
  $currentProvider,
  $selectedStoredSessionId,
  $sessions,
  sessionMatchesStoredId,
  setCurrentBranch,
  setCurrentCwdTransient,
  setCurrentFastMode,
  setCurrentPersonality,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setMessages,
  setSessions,
  setTerminalBackend,
  setTurnStartedAt,
  setWorkspaceCwdOwner,
  setYoloActive
} from '@/store/session'
import { dropSessionState, unbindTileRuntime } from '@/store/session-states'
import { pruneDelegateFallbackSubagents, pruneFinishedSessionSubagents, upsertSubagent } from '@/store/subagents'
import { reportMcpToolResult } from '@/store/suggestion-providers/repair'
import { invalidateSkillSuggestionIndex } from '@/store/suggestion-providers/skill'
import { requestScrollToBottom } from '@/store/thread-scroll'
import { clearActiveSessionTodos } from '@/store/todos'
import { recordToolDiff } from '@/store/tool-diffs'
import { setSessionDraftingTool } from '@/store/tool-drafting'
import { reportInstallMethodWarning } from '@/store/updates'
import { notifyWorkspaceChanged, toolChangedPath, toolMayMutateFiles } from '@/store/workspace-events'
import { ingestBackendSkin } from '@/themes/backend-sync'
import type { RpcEvent } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'
import { finalizeInterruptedMessages } from '../use-prompt-actions/rewind'

import {
  hasSessionInfoStatePatch,
  PRE_TURN_LIVE_SETTLE_GRACE_MS,
  sessionInfoStatePatch,
  SUBAGENT_EVENT_TYPES,
  toTodoPayload
} from './utils'

function firstBillingLine(text: string): string {
  return (text || '').split('\n')[0]?.trim() ?? ''
}

/**
 * Whether a `session.info` payload's `stored_session_id` may be treated as the
 * selected conversation's, so its cwd can be claimed for it (#71254).
 *
 * Absent is not the same as different: the backend omits the id on a
 * not-yet-built (`lazy`) session, and refusing there would leave the workspace
 * marked un-owned for the rest of the conversation. Matching goes through the
 * lineage (`sessionMatchesStoredId`) so a compression-rotated tip and the root
 * a pinned-row selection may hold still read as one conversation.
 */
function sessionInfoDescribesSelectedSession(storedSessionId: string | undefined): boolean {
  const infoStoredSessionId = storedSessionId?.trim() || null
  const selected = $selectedStoredSessionId.get() ?? null

  if (!infoStoredSessionId) {
    return true
  }

  if (!selected) {
    return false
  }

  if (infoStoredSessionId === selected) {
    return true
  }

  return $sessions
    .get()
    .some(session => sessionMatchesStoredId(session, infoStoredSessionId) && sessionMatchesStoredId(session, selected))
}

/**
 * A turn failed on a billing wall (out of credits / payment required). The
 * gateway forwards the structured descriptor built by `agent/billing_links.py`;
 * we cache it per-session (drives the in-chat banner) AND raise one sticky,
 * billing-specific toast — never the generic "Hermes error" — with a smart CTA
 * (Nous → in-app Settings → Billing, other providers → their billing page).
 */
function surfaceBillingBlock(sessionId: string, raw: unknown): void {
  if (!raw || typeof raw !== 'object') {
    return
  }

  const block = raw as BillingBlock

  if (typeof block.provider !== 'string') {
    return
  }

  setBillingBlock(sessionId, block)

  const ctaCopy = {
    addCredits: translateNow('billingBlock.addCredits'),
    openBilling: translateNow('billingBlock.openBilling')
  }

  notify({
    id: `billing-block:${block.provider}`,
    kind: 'warning',
    icon: 'credit-card',
    title: block.is_nous
      ? translateNow('billingBlock.titleNous')
      : translateNow('billingBlock.titleProvider', block.provider_label),
    message: firstBillingLine(block.message) || translateNow('billingBlock.fallbackMessage'),
    durationMs: 0,
    action: { label: billingCtaLabel(block, ctaCopy), onClick: () => runBillingRecovery(block) }
  })
}

/**
 * Events that retire a "drafting a tool call" claim.
 *
 * `tool.generating` opens the claim and nothing closes it — a draft can be
 * abandoned without ever reaching `tool.start`, so enumerating the ways one
 * *ends* left the label on screen for the rest of the turn. Inverted: the
 * claim only covers what the model is emitting right now, and any other output
 * from the session proves it moved on. Same rule the TUI applies to its
 * transient trail lines (`turnController.pruneTransient`).
 */
const DRAFT_SUPERSEDING_EVENT_TYPES = new Set([
  'error',
  'message.complete',
  'message.delta',
  'message.start',
  'reasoning.delta',
  'thinking.delta',
  'tool.complete',
  'tool.progress',
  'tool.start'
])

const COMPACTION_RESUME_EVENT_TYPES = new Set([
  'message.delta',
  'message.interim',
  'thinking.delta',
  'reasoning.delta',
  'reasoning.available',
  'moa.reference',
  'moa.aggregating',
  'moa.progress',
  'moa.phase',
  'tool.start',
  'tool.progress',
  'tool.generating',
  'tool.complete'
])

const PROVIDER_WAIT_SUPERSEDING_EVENT_TYPES = new Set([
  'error',
  'message.complete',
  'message.delta',
  'message.interim',
  'message.start',
  'reasoning.available',
  'reasoning.delta',
  'tool.complete',
  'tool.generating',
  'tool.progress',
  'tool.start'
])

interface GatewayEventDeps {
  activeGatewayProfile: string
  activeSessionIdRef: MutableRefObject<string | null>
  compactedTurnRef: MutableRefObject<Set<string>>
  lastCwdInfoSessionRef: MutableRefObject<string | null>
  nativeSubagentSessionsRef: MutableRefObject<Set<string>>
  appendAssistantDelta: (sessionId: string, delta: string, occurredAt?: number) => void
  appendReasoningDelta: (sessionId: string, delta: string, replace?: boolean, occurredAt?: number) => void
  completeAssistantMessage: (
    sessionId: string,
    text: string,
    responsePreviewed?: boolean,
    failure?: { error: string; partial: boolean },
    occurredAt?: number
  ) => void
  failAssistantMessage: (sessionId: string, errorMessage: string, occurredAt?: number) => void
  flushQueuedDeltas: (sessionId?: string) => void
  finalizeInterimAssistantMessage: (sessionId: string, text: string, occurredAt?: number) => void
  hydrateFromStoredSession: (
    attempts?: number,
    storedSessionId?: string | null,
    runtimeSessionId?: string | null
  ) => Promise<void>
  queryClient: QueryClient
  refreshHermesConfig: () => Promise<void>
  scheduleSessionsRefresh: () => void
  sessionInterrupted: (sessionId: string) => boolean
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
  upsertToolCall: (
    sessionId: string,
    payload: GatewayEventPayload | undefined,
    phase: 'running' | 'complete',
    sourceEventType?: string,
    occurredAt?: number
  ) => void
}

/** The gateway-event dispatcher, extracted from useMessageStream. */
export function useGatewayEventHandler(deps: GatewayEventDeps) {
  const {
    appendAssistantDelta,
    appendReasoningDelta,
    activeGatewayProfile,
    activeSessionIdRef,
    compactedTurnRef,
    lastCwdInfoSessionRef,
    nativeSubagentSessionsRef,
    completeAssistantMessage,
    failAssistantMessage,
    flushQueuedDeltas,
    finalizeInterimAssistantMessage,
    hydrateFromStoredSession,
    queryClient,
    refreshHermesConfig,
    scheduleSessionsRefresh,
    sessionInterrupted,
    sessionStateByRuntimeIdRef,
    updateSessionState,
    upsertToolCall
  } = deps

  const unscopedStreamSessionIdRef = useRef<string | null>(null)

  const configRefreshTimerRef = useRef<null | number>(null)

  const scheduleConfigRefresh = useCallback(() => {
    if (configRefreshTimerRef.current !== null) {
      return
    }

    if (typeof window === 'undefined') {
      void refreshHermesConfig()

      return
    }

    configRefreshTimerRef.current = window.setTimeout(() => {
      configRefreshTimerRef.current = null
      void refreshHermesConfig()
    }, 300)
  }, [refreshHermesConfig])

  useEffect(
    () => () => {
      if (configRefreshTimerRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(configRefreshTimerRef.current)
        configRefreshTimerRef.current = null
      }
    },
    []
  )

  return useCallback(
    (event: RpcEvent) => {
      const payload = event.payload as GatewayEventPayload | undefined

      const fromActiveSource = (): boolean =>
        (!event.profile || normalizeProfileKey(event.profile) === normalizeProfileKey($activeGatewayProfile.get())) &&
        registryBackendScopeKey(event.connectionId ?? null, event.profile ?? null) ===
          registryBackendScopeKey(activeGatewayConnectionId(), event.profile ?? null)

      const occurredAt =
        typeof payload?.timestamp === 'number' && Number.isFinite(payload.timestamp)
          ? payload.timestamp
          : Date.now() / 1000

      const explicitSid = event.session_id || ''

      const route = resolveGatewayEventSessionId({
        activeSessionId: activeSessionIdRef.current,
        eventType: event.type,
        explicitSessionId: explicitSid,
        unscopedStreamSessionId: unscopedStreamSessionIdRef.current
      })

      unscopedStreamSessionIdRef.current = route.nextUnscopedStreamSessionId

      if (route.drop) {
        return
      }

      const sessionId = route.sessionId

      if (
        sessionId &&
        !explicitSid &&
        !route.pinned &&
        event.type &&
        event.type !== 'message.start' &&
        UNSCOPED_STREAM_EVENT_TYPES.has(event.type)
      ) {
        const state = sessionStateByRuntimeIdRef.current.get(sessionId)

        const hasLiveTurn = Boolean(
          state && (state.awaitingResponse || state.busy || state.streamId || state.sawAssistantPayload)
        )

        if (!hasLiveTurn) {
          return
        }
      }

      const isActiveEvent = !!sessionId && sessionId === activeSessionIdRef.current

      const replaySessionId = approvalReplaySessionId(event.type, activeSessionIdRef.current, sessionId)

      if (replaySessionId) {
        void replayPendingApproval($gateway.get(), replaySessionId).catch(() => undefined)
      }

      if (sessionId && COMPACTION_RESUME_EVENT_TYPES.has(event.type) && compactedTurnRef.current.has(sessionId)) {
        setSessionCompacting(sessionId, false)
      }

      if (sessionId && DRAFT_SUPERSEDING_EVENT_TYPES.has(event.type)) {
        setSessionDraftingTool(sessionId, '')
      }

      if (sessionId && PROVIDER_WAIT_SUPERSEDING_EVENT_TYPES.has(event.type)) {
        setSessionProviderWait(sessionId, '')
      }

      if (event.type === 'gateway.ready') {
        ingestBackendSkin((payload as { skin?: HermesSkin } | undefined)?.skin, { apply: false })
        setChangeEventsAvailable(Boolean((payload as { change_events?: boolean } | undefined)?.change_events))

        return
      } else if (event.type === 'skin.changed') {
        if (fromActiveSource()) {
          ingestBackendSkin(payload as HermesSkin | undefined, { apply: true })
        }

        return
      } else if (
        event.type === 'pet.changed' ||
        event.type === 'cron.changed' ||
        event.type === 'sessions.changed' ||
        event.type === 'platforms.changed' ||
        event.type === 'pairing.changed'
      ) {
        if (fromActiveSource()) {
          if (event.type === 'pet.changed') {
            notifyPetChanged(payload as PetChangeMeta | undefined)
          } else if (event.type === 'cron.changed') {
            notifyCronChanged()
          } else if (event.type === 'platforms.changed') {
            notifyPlatformsChanged()
          } else if (event.type === 'pairing.changed') {
            notifyPairingChanged()
          } else {
            notifySessionsChanged()
          }
        }

        return
      } else if (event.type === 'session.reclaimed') {
        const reclaimedRuntimeId = String((payload as { session_id?: string } | undefined)?.session_id ?? '')

        if (reclaimedRuntimeId) {
          dropSessionState(reclaimedRuntimeId)
          unbindTileRuntime(reclaimedRuntimeId)
          sessionStateByRuntimeIdRef.current.delete(reclaimedRuntimeId)
        }

        notifySessionsChanged()

        return
      } else if (event.type === 'session.info') {
        const apply = explicitSid ? isActiveEvent : !activeSessionIdRef.current
        const statePatch = sessionInfoStatePatch(payload)
        const hasStatePatch = hasSessionInfoStatePatch(statePatch)
        const modelChanged = typeof payload?.model === 'string'
        const providerChanged = typeof payload?.provider === 'string'
        const runningChanged = typeof payload?.running === 'boolean'
        const knownState = sessionId ? sessionStateByRuntimeIdRef.current.get(sessionId) : undefined
        const modelValueChanged = modelChanged && payload!.model !== (knownState?.model ?? $currentModel.get())

        const providerValueChanged =
          providerChanged && payload!.provider !== (knownState?.provider ?? $currentProvider.get())

        if (isActiveEvent && typeof payload?.approval_mode === 'string' && event.profile && fromActiveSource()) {
          reconcileApprovalModeForProfile(event.profile, payload.approval_mode)
        }

        if (apply) {

          if (typeof payload?.cwd === 'string' && sessionInfoDescribesSelectedSession(payload.stored_session_id)) {
            const cwdMoved = payload.cwd !== $currentCwd.get()
            const sameSession = !!sessionId && sessionId === lastCwdInfoSessionRef.current

            lastCwdInfoSessionRef.current = sessionId
            setCurrentCwdTransient(payload.cwd)

            setWorkspaceCwdOwner($selectedStoredSessionId.get())

            if (cwdMoved && sameSession) {
              void followActiveSessionCwd(payload.cwd)
            }
          }

          if (typeof payload?.branch === 'string') {
            setCurrentBranch(payload.branch)
          }

          if (typeof payload?.terminal_backend === 'string') {
            setTerminalBackend(payload.terminal_backend)
          }

          if (typeof payload?.personality === 'string') {
            setCurrentPersonality(normalizePersonalityValue(payload.personality))
          }

          if (typeof payload?.reasoning_effort === 'string') {
            setCurrentReasoningEffort(payload.reasoning_effort)
          }

          if (typeof payload?.service_tier === 'string') {
            setCurrentServiceTier(payload.service_tier)
          }

          if (typeof payload?.fast === 'boolean') {
            setCurrentFastMode(payload.fast)
          }

          if (typeof payload?.yolo === 'boolean') {
            setYoloActive(payload.yolo)
          }
        }

        if (sessionId && hasStatePatch) {
          updateSessionState(
            sessionId,
            state => ({
              ...state,
              ...statePatch,
              branch: statePatch.branch ?? state.branch,
              cwd: statePatch.cwd ?? state.cwd
            }),
            payload?.stored_session_id || undefined
          )
        }

        if (runningChanged && sessionId) {
          let recoveredWithoutPayload = false

          const nextState = updateSessionState(
            sessionId,
            state => {
              const busy = Boolean(payload!.running)

              if (state.busy === busy && (busy || !state.awaitingResponse)) {
                return state
              }

              if (busy) {
                if (state.interrupted) {
                  return state
                }

                const gatewayTurnStartedAt =
                  typeof payload!.turn_started_at === 'number' && payload!.turn_started_at > 0
                    ? payload!.turn_started_at * 1000
                    : null

                return {
                  ...state,
                  busy,
                  turnLive: true,
                  turnStartedAt: state.turnStartedAt ?? gatewayTurnStartedAt ?? Date.now()
                }
              }

              const armedAt = state.turnStartedAt

              const withinPreStartGrace =
                typeof armedAt === 'number' && Date.now() - armedAt < PRE_TURN_LIVE_SETTLE_GRACE_MS

              if (state.awaitingResponse && !state.sawAssistantPayload && !state.turnLive && withinPreStartGrace) {
                return state
              }

              recoveredWithoutPayload = state.awaitingResponse && !state.sawAssistantPayload

              return {
                ...state,
                awaitingResponse: false,
                busy,
                messages: finalizeInterruptedMessages(state.messages, state.streamId, occurredAt),
                pendingBranchGroup: null,
                streamId: null,
                turnStartedAt: null,
                turnLive: false
              }
            },
            payload?.stored_session_id || undefined
          )

          if (recoveredWithoutPayload) {
            scheduleSessionsRefresh()

            if (isActiveEvent) {
              void hydrateFromStoredSession(3, nextState.storedSessionId, sessionId)
            }
          }
        }

        if (payload?.usage && (!explicitSid || isActiveEvent)) {
          setCurrentUsage(current => ({ ...current, ...payload.usage }))
        }

        requestDesktopOnboardingForCredentialWarning(payload?.credential_warning)

        if (apply) {
          reportInstallMethodWarning(payload?.install_warning)
          scheduleConfigRefresh()
        }

        if (modelValueChanged || providerValueChanged) {
          void queryClient.invalidateQueries({
            queryKey:
              explicitSid && sessionId ? modelOptionsQueryKey(activeGatewayProfile, sessionId) : ['model-options']
          })
        }
      } else if (event.type === 'session.usage') {
        if (payload?.usage && sessionId) {
          updateSessionState(sessionId, state => ({
            ...state,
            usage: { calls: 0, input: 0, output: 0, total: 0, ...state.usage, ...payload.usage }
          }))

          if (isActiveEvent) {
            setCurrentUsage(current => ({ ...current, ...payload.usage }))
          }
        }
      } else if (event.type === 'message.start') {
        if (!sessionId) {
          return
        }

        flushQueuedDeltas(sessionId)
        pruneFinishedSessionSubagents(sessionId)
        setSessionCompacting(sessionId, false)
        compactedTurnRef.current.delete(sessionId)
        nativeSubagentSessionsRef.current.delete(sessionId)
        clearBillingBlock(sessionId)

        if (isActiveEvent) {
          triggerHaptic('streamStart')
        }

        const seededAt = sessionStateByRuntimeIdRef.current.get(sessionId)?.turnStartedAt

        if (typeof seededAt === 'number') {
          console.debug('[turn-accept-latency]', { sessionId, ms: Date.now() - seededAt })
        }

        updateSessionState(sessionId, state => {
          if (state.interrupted) {
            return state
          }

          return {
            ...state,
            busy: true,
            awaitingResponse: true,
            sawAssistantPayload: false,
            interrupted: false,
            interimBoundaryPending: false,
            turnLive: true,
            turnStartedAt: state.turnStartedAt ?? Date.now()
          }
        })

        if (isActiveEvent) {
          setTurnStartedAt(sessionStateByRuntimeIdRef.current.get(sessionId)?.turnStartedAt ?? Date.now())
        }
      } else if (event.type === 'message.delta') {
        if (sessionId) {
          appendAssistantDelta(sessionId, coerceGatewayText(payload?.text), occurredAt)
        }
      } else if (event.type === 'message.interim') {
        if (sessionId) {
          flushQueuedDeltas(sessionId)
          const text = coerceGatewayText(payload?.text)

          if (text) {
            finalizeInterimAssistantMessage(sessionId, text, occurredAt)
          }
        }
      } else if (event.type === 'thinking.delta') {
        if (sessionId) {
          setSessionProviderWait(sessionId, providerWaitText(coerceGatewayText(payload?.text)))
        }
      } else if (event.type === 'reaction') {
        if (isActiveEvent && (payload?.kind ?? 'vibe') === 'vibe') {
          burstVibeHearts()
        }
      } else if (event.type === 'reasoning.delta') {
        if (sessionId) {
          appendReasoningDelta(sessionId, coerceThinkingText(payload?.text), false, occurredAt)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'reasoning.available') {
        if (sessionId) {
          appendReasoningDelta(sessionId, coerceThinkingText(payload?.text), true, occurredAt)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'moa.reference') {
        if (sessionId) {
          const label = coerceGatewayText(payload?.label) || 'reference'
          const idx = typeof payload?.index === 'number' ? payload.index : undefined
          const cnt = typeof payload?.count === 'number' ? payload.count : undefined
          const header = idx && cnt ? `◇ Reference ${idx}/${cnt} — ${label}` : `◇ Reference — ${label}`
          const body = coerceThinkingText(payload?.text)
          const text = `${header}\n${body}\n\n`

          if (idx === undefined || idx <= 1) {
            appendReasoningDelta(sessionId, text, true, occurredAt)
          } else {
            appendReasoningDelta(sessionId, text, false, occurredAt)
            flushQueuedDeltas(sessionId)
          }
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'moa.aggregating') {
        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'moa.progress') {
        if (sessionId && typeof payload?.refs_done === 'number' && typeof payload?.refs_total === 'number') {
          const label = coerceGatewayText(payload?.label)

          const line = label
            ? `◇ MoA refs ${payload.refs_done}/${payload.refs_total} — ${label}\n`
            : `◇ MoA refs ${payload.refs_done}/${payload.refs_total}\n`

          appendReasoningDelta(sessionId, line, payload.refs_done <= 1, occurredAt)
          flushQueuedDeltas(sessionId)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'moa.phase') {
        if (sessionId && payload?.phase === 'aggregator') {
          appendReasoningDelta(sessionId, '◇ MoA aggregating…\n', false, occurredAt)
          flushQueuedDeltas(sessionId)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'message.complete') {
        if (!sessionId) {
          return
        }

        clearAllPrompts(sessionId)
        clearClarifyRequest(undefined, sessionId)
        clearActiveSessionTodos(sessionId)
        setSessionCompacting(sessionId, false)

        flushQueuedDeltas(sessionId)

        playCompletionSound(sessionId)

        const finalText = coerceGatewayText(payload?.text) || coerceGatewayText(payload?.rendered)

        const failure =
          payload?.status === 'error'
            ? {
                error: coerceGatewayText(payload.error).trim() || finalText || 'Hermes reported an error',
                partial: Boolean(payload.partial)
              }
            : undefined

        completeAssistantMessage(sessionId, finalText, payload?.response_previewed, failure, occurredAt)

        if (payload?.billing) {
          surfaceBillingBlock(sessionId, payload.billing)
        }

        if (isActiveEvent) {
          setTurnStartedAt(null)

          flashPetActivity({ celebrate: true, reasoning: false, toolRunning: false }, 2200)

          if (typeof document !== 'undefined' && !document.hasFocus()) {
            markPetUnread()
          }
        }

        if (payload?.usage) {
          updateSessionState(sessionId, state => ({
            ...state,
            usage: { calls: 0, input: 0, output: 0, total: 0, ...state.usage, ...payload.usage }
          }))

          if (isActiveEvent) {
            setCurrentUsage(current => ({ ...current, ...payload.usage }))
          }
        }
      } else if (event.type === 'session.title') {
        const storedId = typeof payload?.session_id === 'string' ? payload.session_id : ''
        const nextTitle = typeof payload?.title === 'string' ? payload.title.trim() : ''

        if (storedId && nextTitle) {
          setSessions(prev => prev.map(s => (sessionMatchesStoredId(s, storedId) ? { ...s, title: nextTitle } : s)))
        }
      } else if (event.type === 'tool.generating') {
        if (!sessionId || sessionInterrupted(sessionId)) {
          return
        }

        setSessionDraftingTool(sessionId, typeof payload?.name === 'string' ? payload.name : '')

        if (isActiveEvent) {
          setPetActivity({ reasoning: false, toolRunning: true })
        }
      } else if (event.type === 'tool.start' || event.type === 'tool.progress') {
        if (!sessionId) {
          return
        }

        flushQueuedDeltas(sessionId)
        upsertToolCall(sessionId, toTodoPayload(payload) ?? payload, 'running', event.type, occurredAt)

        if (isActiveEvent) {
          setPetActivity({ reasoning: false, toolRunning: true })
        }
      } else if (event.type === 'tool.complete') {
        if (sessionId) {
          flushQueuedDeltas(sessionId)
          upsertToolCall(sessionId, toTodoPayload(payload) ?? payload, 'complete', event.type, occurredAt)

          if (isActiveEvent) {
            setPetActivity({ toolRunning: false })

            if (payload?.error) {
              flashPetActivity({ error: true })
            }
          }

          updateSessionState(sessionId, state => (state.needsInput ? { ...state, needsInput: false } : state))

          if (!sessionInterrupted(sessionId) && (payload?.name === 'terminal' || payload?.name === 'process')) {
            void refreshBackgroundProcesses(sessionId)
          }
        }

        if (payload?.name === 'skill_manage') {
          invalidateSlashCompletions()
          invalidateSkillSuggestionIndex()
        }

        if (sessionId && typeof payload?.name === 'string' && payload.name.startsWith('mcp__')) {
          reportMcpToolResult(
            sessionId,
            payload.name,
            Boolean(payload.error),
            [payload.error, payload.result].filter(part => typeof part === 'string').join(' ')
          )
        }

        if (typeof payload?.inline_diff === 'string' && payload.inline_diff.trim()) {
          recordToolDiff(payload.tool_id || payload.name || '', payload.inline_diff)
        }

        if (payload && toolMayMutateFiles(payload)) {
          notifyWorkspaceChanged(toolChangedPath(payload))
        }
      } else if (SUBAGENT_EVENT_TYPES.has(event.type)) {
        if (sessionId && payload && !sessionInterrupted(sessionId)) {
          if (!nativeSubagentSessionsRef.current.has(sessionId)) {
            pruneDelegateFallbackSubagents(sessionId)
          }

          nativeSubagentSessionsRef.current.add(sessionId)
          upsertSubagent(
            sessionId,
            payload as Record<string, unknown>,
            event.type === 'subagent.spawn_requested' || event.type === 'subagent.start',
            event.type
          )
        }
      } else if (event.type === 'clarify.request') {
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''
        const question = typeof payload?.question === 'string' ? payload.question : ''
        const rawChoices = payload?.choices
        const choices = normalizeChoices(rawChoices)
        const multiSelect = payload?.multi_select === true

        if (requestId && question) {
          if (rawChoices != null && choices.length === 0) {
            warnDroppedChoices('gateway', question, rawChoices)
          }

          setClarifyRequest({
            requestId,
            question,
            choices: choices.length > 0 ? choices : null,
            multiSelect,
            sessionId: sessionId ?? null
          })

          if (sessionId) {
            upsertToolCall(
              sessionId,
              {
                args: { choices, ...(multiSelect ? { multi_select: true } : {}), question },
                name: 'clarify',
                tool_id: requestId
              },
              'running',
              event.type,
              occurredAt
            )

            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))

            if (sessionId === activeSessionIdRef.current) {
              requestScrollToBottom()
            }
          }

          dispatchNativeNotification({
            body: question,
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'mcp.setup.request') {
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''
        const server = typeof payload?.server === 'string' ? payload.server : ''
        const rawAction = typeof payload?.action === 'string' ? payload.action : 'install'
        const action = rawAction === 'enable' || rawAction === 'authorize' ? rawAction : 'install'
        const reason = typeof payload?.reason === 'string' ? payload.reason : ''

        if (requestId && server) {
          setMcpSetupRequest({ action, reason, requestId, server, sessionId: sessionId ?? null })

          if (sessionId) {
            upsertToolCall(
              sessionId,
              { args: { action, reason, server }, name: 'setup_mcp', tool_id: requestId },
              'running'
            )
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: reason || server,
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'approval.request') {
        const command = typeof payload?.command === 'string' ? payload.command : ''
        const description = typeof payload?.description === 'string' ? payload.description : 'dangerous command'

        void receiveApprovalRequest($gateway.get(), {
          allowPermanent: payload?.allow_permanent !== false,
          choices: Array.isArray(payload?.choices)
            ? payload.choices.filter(choice => typeof choice === 'string')
            : undefined,
          command,
          description,
          requestId: typeof payload?.request_id === 'string' ? payload.request_id : undefined,
          sessionId: sessionId ?? null,
          smartDenied: payload?.smart_denied === true
        }).catch(() => undefined)

        if (sessionId) {
          updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
        }

        dispatchNativeNotification({
          actions: [
            { id: 'approve', text: translateNow('notifications.native.approveAction') },
            { id: 'reject', text: translateNow('notifications.native.rejectAction') }
          ],
          body: command || description,
          kind: 'approval',
          sessionId,
          title: translateNow('notifications.native.approvalTitle')
        })
      } else if (event.type === 'sudo.request') {
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          setSudoRequest({ requestId, sessionId: sessionId ?? null })

          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: translateNow('notifications.native.inputBody'),
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'secret.request') {
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const envVar = typeof payload?.env_var === 'string' ? payload.env_var : ''
          const promptText = typeof payload?.prompt === 'string' ? payload.prompt : ''

          setSecretRequest({
            requestId,
            envVar,
            prompt: promptText,
            sessionId: sessionId ?? null
          })

          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: promptText || envVar || translateNow('notifications.native.inputBody'),
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'terminal.read.request') {
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const start = typeof payload?.start === 'number' ? payload.start : undefined
          const count = typeof payload?.count === 'number' ? payload.count : undefined
          const result = readActiveTerminal({ start, count })

          void $gateway.get()?.request('terminal.read.respond', {
            request_id: requestId,
            text: result ? JSON.stringify(result) : ''
          })
        }
      } else if (event.type === 'preview.read.request') {
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const start = typeof payload?.start === 'number' ? payload.start : undefined
          const count = typeof payload?.count === 'number' ? payload.count : undefined

          void readActivePreview({ count, start }).then(result => {
            void $gateway.get()?.request('preview.read.respond', {
              request_id: requestId,
              text: result ? JSON.stringify(result) : ''
            })
          })
        }
      } else if (event.type === 'window.read.request') {
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const read = window.hermesDesktop?.readWindowBelow

          const answer = (result: unknown) =>
            $gateway.get()?.request('window.read.respond', {
              request_id: requestId,
              text: result ? JSON.stringify(result) : ''
            })

          void Promise.resolve(read ? read() : null).then(answer, () => answer(null))
        }
      } else if (event.type === 'agent.terminal.output') {
        writeAgentTerminalChunk(payload?.process_id ?? '', payload?.chunk ?? '')
      } else if (event.type === 'terminal.close') {
        closeAgentTerminalByProc(payload?.process_id ?? '')
      } else if (event.type === 'pane.reveal') {
        if (isActiveEvent) {
          revealDesktopPane(payload?.pane ?? '')
        }
      } else if (event.type === 'message.reaction') {
        const reactedRowId = payload?.row_id

        if (typeof reactedRowId === 'number') {
          const nextReactions = Array.isArray(payload?.reactions) ? payload.reactions : []
          const reactedRole = payload?.role === 'assistant' ? 'assistant' : 'user'

          setMessages(messages => {
            const byRowId = messages.find(message => message.rowId === reactedRowId)

            if (byRowId) {
              recordAgentReaction(reactedRowId, nextReactions)

              return messages.map(message =>
                message.rowId === reactedRowId ? { ...message, reactions: nextReactions } : message
              )
            }

            const lastIndex = messages.findLastIndex(
              message => message.role === reactedRole && message.rowId === undefined
            )

            if (lastIndex === -1) {
              return messages
            }

            recordAgentReaction(reactedRowId, nextReactions)

            return messages.map((message, index) =>
              index === lastIndex ? { ...message, rowId: reactedRowId, reactions: nextReactions } : message
            )
          })
        }
      } else if (event.type === 'status.update') {
        if (sessionId && payload?.kind === 'compacting') {
          setSessionCompacting(sessionId, true)
          compactedTurnRef.current.add(sessionId)
        } else if (sessionId && payload?.kind === 'compacted') {
          setSessionCompacting(sessionId, false)
          compactedTurnRef.current.delete(sessionId)
        } else if (sessionId && payload?.kind === 'process') {
          void refreshBackgroundProcesses(sessionId)
        } else if (sessionId && payload?.kind === 'goal') {
          applyGoalStatusText(sessionId, coerceGatewayText(payload?.text))
        }
      } else if (event.type === 'review.summary') {
        const text = coerceGatewayText(payload?.text)
          .trim()
          .replace(/^[^\p{L}\p{N}]+/u, '')

        if (text && sessionId) {
          flushQueuedDeltas(sessionId)
          updateSessionState(sessionId, state => ({
            ...state,
            messages: [
              ...state.messages,
              {
                id: `review-summary-${Date.now()}`,
                role: 'system',
                parts: [textPart(`review:${text}`, occurredAt)],
                timestamp: occurredAt
              }
            ]
          }))
        }
      } else if (event.type === 'notification.show') {
        const notice = event.payload as AgentNoticePayload | undefined

        showAgentNotice(notice)

        const native = nativeNoticeInput(notice, translateNow('notifications.native.creditsTitle'))

        if (native) {
          dispatchNativeNotification(native)
        }

        if (notice?.key?.startsWith('credits.')) {
          void queryClient.invalidateQueries({ queryKey: ['billing', 'state'] })
        }
      } else if (event.type === 'notification.clear') {
        clearAgentNotice((event.payload as AgentNoticePayload | undefined)?.key)
      } else if (event.type === 'error') {
        const errorMessage = payload?.message || 'Hermes reported an error'
        const looksLikeProviderSetup = isProviderSetupErrorMessage(errorMessage)

        if (sessionId) {
          clearAllPrompts(sessionId)
          clearClarifyRequest(undefined, sessionId)
          clearActiveSessionTodos(sessionId)
          setSessionCompacting(sessionId, false)
          compactedTurnRef.current.delete(sessionId)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: false, toolRunning: false })
          flashPetActivity({ error: true })
        }

        dispatchNativeNotification({
          body: errorMessage,
          kind: 'turnError',
          sessionId,
          title: translateNow('notifications.native.turnErrorTitle')
        })

        if (looksLikeProviderSetup) {
          requestDesktopOnboarding(errorMessage)
        } else if (isDiskFullErrorMessage(errorMessage)) {
          notifyError(new Error(errorMessage), translateNow('notifications.errors.diskFull'))
        } else {
          notify({
            id: `gateway-error:${errorMessage}`,
            kind: 'error',
            title: 'Hermes error',
            message: errorMessage
          })
        }

        if (sessionId) {
          flushQueuedDeltas(sessionId)
          failAssistantMessage(sessionId, errorMessage, occurredAt)
        }

        if (isActiveEvent) {
          setTurnStartedAt(null)
        }
      }
    },
    [
      appendAssistantDelta,
      appendReasoningDelta,
      activeSessionIdRef,
      activeGatewayProfile,
      compactedTurnRef,
      completeAssistantMessage,
      failAssistantMessage,
      finalizeInterimAssistantMessage,
      flushQueuedDeltas,
      hydrateFromStoredSession,
      lastCwdInfoSessionRef,
      nativeSubagentSessionsRef,
      queryClient,
      scheduleConfigRefresh,
      scheduleSessionsRefresh,
      sessionInterrupted,
      sessionStateByRuntimeIdRef,
      updateSessionState,
      upsertToolCall
    ]
  )
}
