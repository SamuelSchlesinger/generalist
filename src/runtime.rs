//! Conversation-runtime primitives shared by the agent loop and TUI.
//!
//! These types are intentionally single-threaded. The terminal reactor and
//! the non-`Send` provider future are polled by one Tokio task, so `Rc` and
//! `RefCell` give us an authoritative prompt queue without a terminal or
//! agent mutex.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashSet;
use std::future::pending;
use std::rc::Rc;
use tokio::sync::watch;

pub type PromptId = u64;

/// When a prompt queued while a turn is active should be delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMode {
    /// Add to the active turn at its next history-valid boundary.
    Steer,
    /// Start a separate turn after the active turn settles.
    FollowUp,
}

impl DeliveryMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Steer => "steer",
            Self::FollowUp => "follow-up",
        }
    }
}

/// A user-visible item waiting for delivery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedPrompt {
    pub id: PromptId,
    pub text: String,
    pub delivery: DeliveryMode,
}

#[derive(Debug, Default)]
struct QueueState {
    next_id: PromptId,
    items: Vec<QueuedPrompt>,
}

/// Cloneable handle to the single authoritative prompt queue.
#[derive(Debug, Clone, Default)]
pub struct PromptQueue {
    inner: Rc<RefCell<QueueState>>,
}

impl PromptQueue {
    pub fn from_saved(items: Vec<QueuedPrompt>) -> Self {
        let mut seen = HashSet::new();
        let items = items
            .into_iter()
            .filter(|item| !item.text.trim().is_empty() && seen.insert(item.id))
            .collect::<Vec<_>>();
        let next_id = items
            .iter()
            .map(|item| item.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        Self {
            inner: Rc::new(RefCell::new(QueueState { next_id, items })),
        }
    }

    pub fn enqueue(&self, text: impl Into<String>, delivery: DeliveryMode) -> PromptId {
        let mut state = self.inner.borrow_mut();
        let id = state.next_id;
        state.next_id = state.next_id.saturating_add(1);
        state.items.push(QueuedPrompt {
            id,
            text: text.into(),
            delivery,
        });
        id
    }

    pub fn snapshot(&self) -> Vec<QueuedPrompt> {
        self.inner.borrow().items.clone()
    }

    pub fn len(&self) -> usize {
        self.inner.borrow().items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.borrow().items.is_empty()
    }

    pub fn replace(&self, items: Vec<QueuedPrompt>) {
        let replacement = Self::from_saved(items);
        let mut state = self.inner.borrow_mut();
        let replacement = replacement.inner.borrow();
        state.next_id = replacement.next_id;
        state.items.clone_from(&replacement.items);
    }

    pub fn edit(&self, id: PromptId, text: String) -> bool {
        if text.trim().is_empty() {
            return false;
        }
        let mut state = self.inner.borrow_mut();
        let Some(item) = state.items.iter_mut().find(|item| item.id == id) else {
            return false;
        };
        item.text = text;
        true
    }

    pub fn delete(&self, id: PromptId) -> bool {
        let mut state = self.inner.borrow_mut();
        let Some(index) = state.items.iter().position(|item| item.id == id) else {
            return false;
        };
        state.items.remove(index);
        true
    }

    pub fn toggle_delivery(&self, id: PromptId, turn_active: bool) -> bool {
        let mut state = self.inner.borrow_mut();
        let Some(item) = state.items.iter_mut().find(|item| item.id == id) else {
            return false;
        };
        item.delivery = match item.delivery {
            DeliveryMode::Steer => DeliveryMode::FollowUp,
            DeliveryMode::FollowUp if turn_active => DeliveryMode::Steer,
            DeliveryMode::FollowUp => return false,
        };
        true
    }

    pub fn move_by(&self, id: PromptId, offset: isize) -> bool {
        let mut state = self.inner.borrow_mut();
        let Some(index) = state.items.iter().position(|item| item.id == id) else {
            return false;
        };
        let target = index
            .saturating_add_signed(offset)
            .min(state.items.len() - 1);
        if target == index {
            return false;
        }
        let item = state.items.remove(index);
        state.items.insert(target, item);
        true
    }

    /// Remove the latest visible item so it can be restored to the composer.
    pub fn restore_latest(&self) -> Option<QueuedPrompt> {
        self.inner.borrow_mut().items.pop()
    }

    /// Claim the first follow-up. The caller must commit after recording it;
    /// dropping the claim puts it back at the front.
    pub fn claim_follow_up(&self) -> Option<PromptClaim> {
        let mut state = self.inner.borrow_mut();
        let item = state.items.first()?;
        if item.delivery != DeliveryMode::FollowUp {
            return None;
        }
        let item = state.items.remove(0);
        Some(PromptClaim::new(self.clone(), vec![item]))
    }

    /// Atomically claim every steer currently waiting, preserving relative
    /// order and leaving all follow-ups in place.
    pub fn claim_steering(&self) -> Option<PromptClaim> {
        let mut state = self.inner.borrow_mut();
        let mut claimed = Vec::new();
        let mut retained = Vec::with_capacity(state.items.len());
        for item in state.items.drain(..) {
            if item.delivery == DeliveryMode::Steer {
                claimed.push(item);
            } else {
                retained.push(item);
            }
        }
        state.items = retained;
        (!claimed.is_empty()).then(|| PromptClaim::new(self.clone(), claimed))
    }

    /// A steer whose target turn has ended becomes an ordinary follow-up.
    pub fn normalize_steers(&self) {
        for item in &mut self.inner.borrow_mut().items {
            if item.delivery == DeliveryMode::Steer {
                item.delivery = DeliveryMode::FollowUp;
            }
        }
    }

    fn requeue_front(&self, mut items: Vec<QueuedPrompt>) {
        let mut state = self.inner.borrow_mut();
        let visible = state
            .items
            .iter()
            .map(|item| item.id)
            .collect::<HashSet<_>>();
        items.retain(|item| !visible.contains(&item.id));
        items.append(&mut state.items);
        state.items = items;
    }
}

/// A two-phase queue claim. Uncommitted claims roll back on drop.
#[derive(Debug)]
pub struct PromptClaim {
    queue: PromptQueue,
    items: Option<Vec<QueuedPrompt>>,
}

impl PromptClaim {
    fn new(queue: PromptQueue, items: Vec<QueuedPrompt>) -> Self {
        Self {
            queue,
            items: Some(items),
        }
    }

    pub fn prompts(&self) -> &[QueuedPrompt] {
        self.items.as_deref().unwrap_or_default()
    }

    pub fn commit(mut self) -> Vec<QueuedPrompt> {
        self.items.take().unwrap_or_default()
    }
}

impl Drop for PromptClaim {
    fn drop(&mut self) {
        if let Some(items) = self.items.take() {
            self.queue.requeue_front(items);
        }
    }
}

/// Controller-side cancellation handle.
#[derive(Debug, Clone)]
pub struct CancelHandle {
    sender: watch::Sender<bool>,
}

impl CancelHandle {
    pub fn cancel(&self) {
        self.sender.send_replace(true);
    }
}

/// Agent-side turn controls. A detached control preserves the legacy library
/// API; the TUI supplies a real cancellation receiver and prompt queue.
#[derive(Debug)]
pub struct TurnControl {
    cancellation: Option<watch::Receiver<bool>>,
    queue: PromptQueue,
}

impl TurnControl {
    pub fn detached() -> Self {
        Self {
            cancellation: None,
            queue: PromptQueue::default(),
        }
    }

    pub fn for_turn(queue: PromptQueue) -> (CancelHandle, Self) {
        let (sender, receiver) = watch::channel(false);
        (
            CancelHandle { sender },
            Self {
                cancellation: Some(receiver),
                queue,
            },
        )
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(|receiver| *receiver.borrow())
    }

    pub async fn cancelled(&mut self) {
        let Some(receiver) = self.cancellation.as_mut() else {
            pending::<()>().await;
            return;
        };
        if *receiver.borrow() {
            return;
        }
        let _ = receiver.changed().await;
    }

    pub fn claim_steering(&self) -> Option<PromptClaim> {
        self.queue.claim_steering()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_text_keeps_distinct_stable_ids() {
        let queue = PromptQueue::default();
        let first = queue.enqueue("same", DeliveryMode::FollowUp);
        let second = queue.enqueue("same", DeliveryMode::FollowUp);
        assert_ne!(first, second);
        assert_eq!(queue.snapshot().len(), 2);
    }

    #[test]
    fn dropped_claim_rolls_back_and_commit_removes() {
        let queue = PromptQueue::default();
        let first = queue.enqueue("first", DeliveryMode::FollowUp);
        {
            let claim = queue.claim_follow_up().expect("claim");
            assert_eq!(claim.prompts()[0].id, first);
            assert!(queue.is_empty());
        }
        assert_eq!(queue.snapshot()[0].id, first);

        let claim = queue.claim_follow_up().expect("claim");
        let committed = claim.commit();
        assert_eq!(committed[0].id, first);
        assert!(queue.is_empty());
    }

    #[test]
    fn steering_claim_preserves_relative_order_and_followups() {
        let queue = PromptQueue::default();
        let steer_a = queue.enqueue("a", DeliveryMode::Steer);
        let follow = queue.enqueue("b", DeliveryMode::FollowUp);
        let steer_c = queue.enqueue("c", DeliveryMode::Steer);

        let claim = queue.claim_steering().expect("steers");
        assert_eq!(
            claim
                .prompts()
                .iter()
                .map(|prompt| prompt.id)
                .collect::<Vec<_>>(),
            vec![steer_a, steer_c]
        );
        assert_eq!(queue.snapshot()[0].id, follow);
        claim.commit();
    }

    #[test]
    fn dropped_steering_claim_restores_the_same_ids_at_the_front() {
        let queue = PromptQueue::default();
        let first = queue.enqueue("steer one", DeliveryMode::Steer);
        let follow = queue.enqueue("later", DeliveryMode::FollowUp);
        let second = queue.enqueue("steer two", DeliveryMode::Steer);

        {
            let claim = queue.claim_steering().expect("steering claim");
            assert_eq!(
                claim
                    .prompts()
                    .iter()
                    .map(|prompt| prompt.id)
                    .collect::<Vec<_>>(),
                vec![first, second]
            );
        }

        assert_eq!(
            queue
                .snapshot()
                .iter()
                .map(|prompt| prompt.id)
                .collect::<Vec<_>>(),
            vec![first, second, follow]
        );
    }

    #[test]
    fn ending_turn_normalizes_undelivered_steers() {
        let queue = PromptQueue::default();
        queue.enqueue("late", DeliveryMode::Steer);
        queue.normalize_steers();
        assert_eq!(queue.snapshot()[0].delivery, DeliveryMode::FollowUp);
        assert!(queue.claim_follow_up().is_some());
    }

    #[test]
    fn followups_dispatch_one_at_a_time_in_fifo_order() {
        let queue = PromptQueue::default();
        let first = queue.enqueue("first", DeliveryMode::FollowUp);
        let second = queue.enqueue("second", DeliveryMode::FollowUp);

        assert_eq!(queue.claim_follow_up().unwrap().commit()[0].id, first);
        assert_eq!(queue.claim_follow_up().unwrap().commit()[0].id, second);
        assert!(queue.is_empty());
    }

    #[test]
    fn queue_manager_mutations_address_stable_ids() {
        let queue = PromptQueue::default();
        let first = queue.enqueue("first", DeliveryMode::FollowUp);
        let second = queue.enqueue("second", DeliveryMode::FollowUp);
        let third = queue.enqueue("third", DeliveryMode::FollowUp);

        assert!(queue.edit(second, "edited".into()));
        assert!(queue.toggle_delivery(second, true));
        assert!(queue.move_by(third, -2));
        assert!(queue.delete(first));

        let items = queue.snapshot();
        assert_eq!(
            items.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![third, second]
        );
        assert_eq!(items[1].text, "edited");
        assert_eq!(items[1].delivery, DeliveryMode::Steer);
        assert!(!queue.toggle_delivery(third, false));
    }

    #[test]
    fn loading_discards_empty_text_and_duplicate_visible_ids() {
        let queue = PromptQueue::from_saved(vec![
            QueuedPrompt {
                id: 9,
                text: "kept".into(),
                delivery: DeliveryMode::FollowUp,
            },
            QueuedPrompt {
                id: 9,
                text: "duplicate".into(),
                delivery: DeliveryMode::Steer,
            },
            QueuedPrompt {
                id: 10,
                text: "  ".into(),
                delivery: DeliveryMode::FollowUp,
            },
        ]);

        assert_eq!(queue.snapshot().len(), 1);
        assert_eq!(queue.snapshot()[0].text, "kept");
        assert_ne!(queue.enqueue("new", DeliveryMode::FollowUp), 9);
    }
}
