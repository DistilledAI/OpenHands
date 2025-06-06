import { AgentState } from "../agent-state";
import ObservationType from "../observation-type";
import { OpenHandsObservationEvent } from "./base";

export interface AgentStateChangeObservation
  extends OpenHandsObservationEvent<"agent_state_changed"> {
  source: "agent";
  extras: {
    agent_state: AgentState;
  };
}

export interface CommandObservation extends OpenHandsObservationEvent<"run"> {
  source: "agent";
  extras: {
    command: string;
    hidden?: boolean;
    metadata: Record<string, unknown>;
  };
}

export interface IPythonObservation
  extends OpenHandsObservationEvent<"run_ipython"> {
  source: "agent";
  extras: {
    code: string;
  };
}

export interface DelegateObservation
  extends OpenHandsObservationEvent<"delegate"> {
  source: "agent";
  extras: {
    outputs: Record<string, unknown>;
  };
}

export interface BrowseObservation extends OpenHandsObservationEvent<"browse"> {
  source: "agent";
  extras: {
    url: string;
    screenshot: string;
    error: boolean;
    open_page_urls: string[];
    active_page_index: number;
    dom_object: Record<string, unknown>;
    axtree_object: Record<string, unknown>;
    extra_element_properties: Record<string, unknown>;
    last_browser_action: string;
    last_browser_action_error: unknown;
    focused_element_bid: string;
  };
}

export interface BrowseInteractiveObservation
  extends OpenHandsObservationEvent<"browse_interactive"> {
  source: "agent";
  extras: {
    url: string;
    screenshot: string;
    error: boolean;
    open_page_urls: string[];
    active_page_index: number;
    dom_object: Record<string, unknown>;
    axtree_object: Record<string, unknown>;
    extra_element_properties: Record<string, unknown>;
    last_browser_action: string;
    last_browser_action_error: unknown;
    focused_element_bid: string;
  };
}

export interface WriteObservation extends OpenHandsObservationEvent<"write"> {
  source: "agent";
  extras: {
    path: string;
    content: string;
  };
}

export interface ReadObservation extends OpenHandsObservationEvent<"read"> {
  source: "agent";
  extras: {
    path: string;
    impl_source: string;
  };
}

export interface EditObservation extends OpenHandsObservationEvent<"edit"> {
  source: "agent";
  extras: {
    path: string;
    diff: string;
    impl_source: string;
  };
}

export interface ErrorObservation extends OpenHandsObservationEvent<"error"> {
  source: "user";
  extras: {
    error_id?: string;
  };
}

export interface AgentThinkObservation
  extends OpenHandsObservationEvent<"think"> {
  source: "agent";
  extras: {
    thought: string;
  };
}

export interface PlaywrightMcpBrowserScreenshotObservation
  extends OpenHandsObservationEvent<ObservationType.PLAYWRIGHT_MCP_BROWSER_SCREENSHOT> {
  source: "agent";
  extras: {
    url: string;
    screenshot: string;
    trigger_by_action: string;
  };
}
export type OpenHandsObservation =
  | AgentStateChangeObservation
  | AgentThinkObservation
  | CommandObservation
  | IPythonObservation
  | DelegateObservation
  | BrowseObservation
  | BrowseInteractiveObservation
  | WriteObservation
  | ReadObservation
  | EditObservation
  | ErrorObservation
  | PlaywrightMcpBrowserScreenshotObservation;
