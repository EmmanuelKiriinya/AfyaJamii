/**
 * React Query hooks for the API.
 *
 * The QueryClientProvider was already mounted but nothing used it — every
 * component fetched in its own useEffect, so the overview and the history view
 * would each hit the API separately and could disagree about what the latest
 * reading was. Going through one cache keeps them consistent and means
 * submitting a reading refreshes everything that depends on it.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type ConversationRecord,
  type VitalsRecord,
  type VitalsSubmitPayload,
} from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";

export const queryKeys = {
  vitals: (limit: number) => ["vitals", limit] as const,
  conversations: (limit: number) => ["conversations", limit] as const,
};

export function useVitalsHistory(limit = 20) {
  const { token } = useAuth();

  return useQuery<VitalsRecord[]>({
    queryKey: queryKeys.vitals(limit),
    queryFn: () => api.getVitalsHistory(token as string, limit),
    enabled: Boolean(token),
  });
}

export function useConversations(limit = 20) {
  const { token } = useAuth();

  return useQuery<ConversationRecord[]>({
    queryKey: queryKeys.conversations(limit),
    queryFn: () => api.getConversationsHistory(token as string, limit),
    enabled: Boolean(token),
  });
}

export function useSubmitVitals() {
  const { token } = useAuth();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: VitalsSubmitPayload) =>
      api.submitVitals(payload, token as string),
    onSuccess: () => {
      // A new reading changes both the history and the conversation log.
      void queryClient.invalidateQueries({ queryKey: ["vitals"] });
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
  });
}

export function useAskQuestion() {
  const { token } = useAuth();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (question: string) => api.chatAdvice(question, token as string),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
  });
}
