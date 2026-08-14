//! Bounded buffering between the agent's event callback and the TUI
//! reactor: stream previews are coalesced so provider chunking cannot
//! amplify the pending queue.

use generalist::AgentEvent;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use tokio::sync::Notify;

pub(crate) const MAX_PENDING_STREAM_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StreamPreviewKind {
    Text,
    Reasoning,
}

#[derive(Debug, Default)]
pub(crate) struct PendingStreamPreview {
    text: String,
    reasoning: String,
    omitted_text_bytes: usize,
    omitted_reasoning_bytes: usize,
    first: Option<StreamPreviewKind>,
}

impl PendingStreamPreview {
    pub(crate) fn append(&mut self, kind: StreamPreviewKind, fragment: String) {
        self.first.get_or_insert(kind);
        let available = MAX_PENDING_STREAM_BYTES.saturating_sub(self.retained_bytes());
        let mut keep = available.min(fragment.len());
        while !fragment.is_char_boundary(keep) {
            keep -= 1;
        }

        let omitted = fragment.len() - keep;
        match kind {
            StreamPreviewKind::Text => {
                self.text.push_str(&fragment[..keep]);
                self.omitted_text_bytes = self.omitted_text_bytes.saturating_add(omitted);
            }
            StreamPreviewKind::Reasoning => {
                self.reasoning.push_str(&fragment[..keep]);
                self.omitted_reasoning_bytes = self.omitted_reasoning_bytes.saturating_add(omitted);
            }
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.text.len() + self.reasoning.len()
    }

    pub(crate) fn into_events(self) -> Vec<AgentEvent> {
        let Self {
            text,
            reasoning,
            omitted_text_bytes,
            omitted_reasoning_bytes,
            first,
        } = self;
        let mut events = Vec::with_capacity(3);
        match first {
            Some(StreamPreviewKind::Text) => {
                if !text.is_empty() {
                    events.push(AgentEvent::AssistantTextDelta(text));
                }
                if !reasoning.is_empty() {
                    events.push(AgentEvent::ReasoningDelta(reasoning));
                }
            }
            Some(StreamPreviewKind::Reasoning) => {
                if !reasoning.is_empty() {
                    events.push(AgentEvent::ReasoningDelta(reasoning));
                }
                if !text.is_empty() {
                    events.push(AgentEvent::AssistantTextDelta(text));
                }
            }
            None => {}
        }
        if omitted_text_bytes > 0 || omitted_reasoning_bytes > 0 {
            events.push(AgentEvent::StreamDisplayTruncated {
                text_bytes: omitted_text_bytes,
                reasoning_bytes: omitted_reasoning_bytes,
            });
        }
        events
    }
}

#[derive(Debug)]
pub(crate) enum BufferedAgentEvent {
    Event(AgentEvent),
    Stream(PendingStreamPreview),
}

#[derive(Clone)]
pub(crate) struct AgentDisplaySender {
    queue: Rc<RefCell<VecDeque<BufferedAgentEvent>>>,
    notify: Rc<Notify>,
}

pub(crate) struct AgentDisplayReceiver {
    queue: Rc<RefCell<VecDeque<BufferedAgentEvent>>>,
    notify: Rc<Notify>,
}

pub(crate) fn agent_display_channel() -> (AgentDisplaySender, AgentDisplayReceiver) {
    let queue = Rc::new(RefCell::new(VecDeque::new()));
    let notify = Rc::new(Notify::new());
    (
        AgentDisplaySender {
            queue: Rc::clone(&queue),
            notify: Rc::clone(&notify),
        },
        AgentDisplayReceiver { queue, notify },
    )
}

impl AgentDisplaySender {
    pub(crate) fn send(&self, event: AgentEvent) {
        let mut queue = self.queue.borrow_mut();
        match event {
            AgentEvent::AssistantTextDelta(text) => {
                Self::append_delta(&mut queue, StreamPreviewKind::Text, text)
            }
            AgentEvent::ReasoningDelta(reasoning) => {
                Self::append_delta(&mut queue, StreamPreviewKind::Reasoning, reasoning)
            }
            event => queue.push_back(BufferedAgentEvent::Event(event)),
        }
        drop(queue);
        self.notify.notify_one();
    }

    pub(crate) fn append_delta(
        queue: &mut VecDeque<BufferedAgentEvent>,
        kind: StreamPreviewKind,
        fragment: String,
    ) {
        if !matches!(queue.back(), Some(BufferedAgentEvent::Stream(_))) {
            queue.push_back(BufferedAgentEvent::Stream(PendingStreamPreview::default()));
        }
        let Some(BufferedAgentEvent::Stream(preview)) = queue.back_mut() else {
            unreachable!("a stream preview was just appended")
        };
        preview.append(kind, fragment);
    }
}

impl AgentDisplayReceiver {
    pub(crate) async fn recv_batch(&self) -> Vec<AgentEvent> {
        loop {
            let notify = Rc::clone(&self.notify);
            let notified = notify.notified();
            if let Some(events) = self.try_recv_batch() {
                return events;
            }
            notified.await;
        }
    }

    pub(crate) fn try_recv_batch(&self) -> Option<Vec<AgentEvent>> {
        self.queue
            .borrow_mut()
            .pop_front()
            .map(|event| match event {
                BufferedAgentEvent::Event(event) => vec![event],
                BufferedAgentEvent::Stream(preview) => preview.into_events(),
            })
    }

    #[cfg(test)]
    pub(crate) fn buffered_records(&self) -> usize {
        self.queue.borrow().len()
    }

    #[cfg(test)]
    pub(crate) fn pending_preview_bytes(&self) -> usize {
        self.queue
            .borrow()
            .iter()
            .filter_map(|event| match event {
                BufferedAgentEvent::Stream(preview) => Some(preview.retained_bytes()),
                BufferedAgentEvent::Event(_) => None,
            })
            .sum()
    }
}
