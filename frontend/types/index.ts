export interface Source {
  doc: string;
  section: string;
  url: string;
  type: string;
  text?: string;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  status?: string;
  loading?: boolean;
  error?: boolean;
  sources?: Source[];
}