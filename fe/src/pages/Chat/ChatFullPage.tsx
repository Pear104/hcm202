import React, { useEffect, useRef, useState } from "react";
import { BiSolidBot } from "react-icons/bi";
import { IoSend, IoSparkles } from "react-icons/io5";
import { FaUser } from "react-icons/fa";
import { marked } from "marked";

type MessageItemProps = {
  content: string;
  sender: "me" | "model";
  timestamp?: Date;
};

type SuggestionItemProps = {
  title: string;
  icon: string;
};

// Gợi ý theo TTHCM
const suggestions: SuggestionItemProps[] = [
  { title: "Khái quát tư tưởng Hồ Chí Minh về đạo đức cách mạng", icon: "🌿" },
  { title: "Vai trò của đại đoàn kết dân tộc trong tư tưởng HCM", icon: "🤝" },
  { title: "Nhà nước của dân, do dân, vì dân — nội hàm cốt lõi", icon: "🏛️" },
  { title: "Giá trị thời đại của tư tưởng Hồ Chí Minh", icon: "✨" },
];

const STORAGE_KEY = "hcm_chat_messages_v1";
const LEGACY_KEYS = [
  "hcm_chat_popup_messages_v1",
  "hcm_chat_fullpage_messages_v1",
];
const MAX_HISTORY = 20;

export default function ChatFullPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const messageEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<MessageItemProps[]>([]);

  function migrateAndLoadHistory(): MessageItemProps[] | null {
    try {
      // đọc khóa mới trước
      const rawNew = localStorage.getItem(STORAGE_KEY);
      const parseArr = (raw: string | null) => {
        if (!raw) return [] as MessageItemProps[];
        const arr = JSON.parse(raw) as Array<
          Omit<MessageItemProps, "timestamp"> & { timestamp?: string }
        >;
        return arr.map((m) => ({
          ...m,
          timestamp: m.timestamp ? new Date(m.timestamp) : undefined,
        })) as MessageItemProps[];
      };

      let merged = parseArr(rawNew);

      // nếu khóa mới chưa có gì, thử gom từ legacy keys
      if (!merged.length) {
        for (const k of LEGACY_KEYS) {
          const raw = localStorage.getItem(k);
          const items = parseArr(raw);
          if (items.length) merged = merged.concat(items);
        }
        // sắp theo thời gian và cắt 20 cuối
        merged.sort((a, b) => {
          const ta = a.timestamp ? a.timestamp.getTime() : 0;
          const tb = b.timestamp ? b.timestamp.getTime() : 0;
          return ta - tb;
        });
        merged = merged.slice(-MAX_HISTORY);
        if (merged.length) {
          localStorage.setItem(
            STORAGE_KEY,
            JSON.stringify(
              merged.map((m) => ({
                ...m,
                timestamp: m.timestamp ? m.timestamp.toISOString() : undefined,
              }))
            )
          );
        }
      }
      return merged.length ? merged : null;
    } catch {
      return null;
    }
  }

  function saveHistory(arr: MessageItemProps[]) {
    const limited = arr.slice(-MAX_HISTORY);
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(
        limited.map((m) => ({
          ...m,
          timestamp: m.timestamp ? m.timestamp.toISOString() : undefined,
        }))
      )
    );
  }

  // Init
  useEffect(() => {
    const restored = migrateAndLoadHistory();
    if (restored && restored.length) {
      setMessages(restored);
    } else {
      setMessages([
        {
          content:
            "Xin chào! 👋 Mình là **Gia sư Tư tưởng Hồ Chí Minh**. Hãy đặt câu hỏi về các luận điểm cốt lõi hoặc các câu chuyện/tiểu sử về Bác nhé.",
          sender: "model",
          timestamp: new Date(),
        },
      ]);
    }
  }, []);

  useEffect(() => {
    if (messages.length) saveHistory(messages);
  }, [messages]);

  // Auto-scroll khi ở gần đáy
  useEffect(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    const threshold = 160;
    const atBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
    if (atBottom && messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading]);

  const handleSendMessage = async (text: string) => {
    if (text.trim() === "") return;

    const meMsg: MessageItemProps = {
      content: text,
      sender: "me",
      timestamp: new Date(),
    };
    setInputValue("");
    setMessages((prev) => [...prev, meMsg]);
    setIsLoading(true);

    try {
      const answer = await getAnswer(text);
      const botMsg: MessageItemProps = {
        content: answer || "Xin lỗi, mình chưa có câu trả lời phù hợp.",
        sender: "model",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (e) {
      console.error(e);
      setMessages((prev) => [
        ...prev,
        {
          content: "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau.",
          sender: "model",
          timestamp: new Date(),
        },
      ]);
    }
    setIsLoading(false);
  };

  const handleSuggestionClick = (s: string) => handleSendMessage(s);

  return (
    <div
      data-lenis-prevent-wheel
      data-lenis-prevent-touch
      className="h-[94vh] min-h-0 bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 flex flex-col overflow-hidden"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-yellow-500 to-yellow-600 px-8 py-6 shadow-lg flex-shrink-0">
        <div className="max-w-5xl mx-auto flex items-center gap-4">
          <div className="w-14 h-14 bg-white rounded-full flex items-center justify-center shadow-lg">
            <BiSolidBot className="text-yellow-600 text-3xl" />
          </div>
          <div>
            <h1 className="font-bold text-white text-2xl">
              Gia sư Tư tưởng Hồ Chí Minh
            </h1>
            <p className="text-yellow-100 text-sm flex items-center gap-2">
              <IoSparkles className="text-yellow-200" />
              Trợ lý
            </p>
          </div>
        </div>
      </div>
      <div className="flex-1 min-h-0 flex flex-col">
        {/* Messages */}
        <div
          ref={messagesContainerRef}
          className="flex-1 overflow-y-auto px-8 py-8 no-scrollbar"
          style={{
            scrollBehavior: "smooth",
            overscrollBehavior: "contain" as const,
          }}
          data-lenis-prevent-wheel
          data-lenis-prevent-touch
          onWheel={(e) => {
            e.preventDefault();
            e.stopPropagation();
            messagesContainerRef.current!.scrollTop += e.deltaY;
          }}
          onWheelCapture={(e) => e.stopPropagation()}
          onTouchMoveCapture={(e) => e.stopPropagation()}
        >
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.length === 1 && messages[0].sender === "model" && (
              <div className="mb-8">
                <h2 className="text-zinc-400 text-lg mb-4 text-center">
                  Câu hỏi gợi ý
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {suggestions.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => handleSuggestionClick(s.title)}
                      className="p-6 bg-zinc-800 hover:bg-zinc-700 rounded-2xl text-left transition-all border border-zinc-700 hover:border-yellow-500/50 group"
                    >
                      <div className="text-4xl mb-3">{s.icon}</div>
                      <div className="text-zinc-300 group-hover:text-yellow-400 transition-colors">
                        {s.title}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <MessageItem key={i} data={m} />
            ))}

            {isLoading && <TypingIndicator />}
            <div ref={messageEndRef} />
          </div>
        </div>
      </div>
      {/* Input */}
      <div className="bg-zinc-900 border-t border-zinc-800 px-8 py-6 flex-shrink-0">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-4 items-end">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              className="flex-1 bg-zinc-800 text-zinc-200 rounded-2xl px-6 py-4 resize-none outline-none focus:ring-2 focus:ring-yellow-500 text-base max-h-40"
              placeholder="Nhập câu hỏi của bạn ..."
              rows={2}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(inputValue);
                }
              }}
            />
            <button
              onClick={() => handleSendMessage(inputValue)}
              disabled={isLoading || inputValue.trim() === ""}
              className="bg-yellow-500 hover:bg-yellow-600 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white px-8 py-4 rounded-2xl transition-all font-medium"
            >
              <IoSend className="text-2xl" />
            </button>
          </div>
          <div className="text-sm text-zinc-500 mt-3 text-center">
            Nhấn Enter để gửi • Shift + Enter để xuống dòng
          </div>
        </div>
      </div>

      <style>{`
        .message-content p { margin-bottom: 0.5rem; }
        .message-content p:last-child { margin-bottom: 0; }
        .message-content strong { font-weight: 600; color: inherit; }
        .message-content ul, .message-content ol { margin-left: 1.5rem; margin-top: 0.5rem; margin-bottom: 0.5rem; }
        .message-content li { margin-bottom: 0.25rem; }
        .message-content code { background-color: rgba(0,0,0,0.3); padding: 0.125rem 0.25rem; border-radius: 0.25rem; font-size: 0.875em; }
        .overflow-y-auto::-webkit-scrollbar { width: 8px; }
        .overflow-y-auto::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); border-radius: 10px; }
        .overflow-y-auto::-webkit-scrollbar-thumb { background: rgba(234,179,8,0.5); border-radius: 10px; }
        .overflow-y-auto::-webkit-scrollbar-thumb:hover { background: rgba(234,179,8,0.7); }
      `}</style>
    </div>
  );
}

// ===== Message item
const MessageItem = ({ data }: { data: MessageItemProps }) => {
  const normalized = data.content
    .replaceAll("Sources:", "**Nguồn:**")
    .replaceAll("GT", "Giáo trình Tư tưởng Hồ Chí Minh")
    .replaceAll("ToanTap", "Hồ Chí Minh Toàn tập")
    .replaceAll("BupSenXanh", "Búp Sen Xanh (Sơn Tùng)");
  const parsedContent = marked(normalized);

  return (
    <div
      className={`flex gap-4 ${
        data.sender === "me" ? "justify-end" : "justify-start"
      }`}
    >
      {data.sender === "model" && (
        <div className="w-12 h-12 bg-yellow-500 rounded-full flex items-center justify-center flex-shrink-0">
          <BiSolidBot className="text-white text-2xl" />
        </div>
      )}
      <div
        className={`max-w-[70%] px-6 py-4 rounded-3xl ${
          data.sender === "me"
            ? "bg-yellow-600 text-white rounded-tr-md"
            : "bg-zinc-800 text-zinc-200 rounded-tl-md"
        }`}
        style={{
          wordWrap: "break-word",
          overflowWrap: "break-word",
          wordBreak: "break-word",
        }}
      >
        <div
          className="message-content text-base leading-relaxed"
          dangerouslySetInnerHTML={{ __html: parsedContent }}
        />
        {data.timestamp && (
          <div className="text-xs opacity-60 mt-2">
            {new Date(data.timestamp).toLocaleTimeString("vi-VN", {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        )}
      </div>
      {data.sender === "me" && (
        <div className="w-12 h-12 bg-zinc-700 rounded-full flex items-center justify-center flex-shrink-0">
          <FaUser className="text-white text-lg" />
        </div>
      )}
    </div>
  );
};

// ===== Typing Indicator
const TypingIndicator = () => (
  <div className="flex gap-4 justify-start">
    <div className="w-12 h-12 bg-yellow-500 rounded-full flex items-center justify-center flex-shrink-0">
      <BiSolidBot className="text-white text-2xl" />
    </div>
    <div className="bg-zinc-800 px-6 py-4 rounded-3xl rounded-tl-md">
      <div className="flex gap-2">
        <div
          className="w-2.5 h-2.5 bg-zinc-600 rounded-full animate-bounce"
          style={{ animationDelay: "0ms" }}
        />
        <div
          className="w-2.5 h-2.5 bg-zinc-600 rounded-full animate-bounce"
          style={{ animationDelay: "150ms" }}
        />
        <div
          className="w-2.5 h-2.5 bg-zinc-600 rounded-full animate-bounce"
          style={{ animationDelay: "300ms" }}
        />
      </div>
    </div>
  </div>
);

// ===== API
const getAnswer = async (question: string) => {
  const body = { question, model_name: "gpt" };
  const response = await fetch(`https://hcm.jangkuz.io.vn/chat`, {
    body: JSON.stringify(body),
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  return (await response.json())?.answer;
};
