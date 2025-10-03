import React, { useEffect, useRef, useState } from "react";
import { BiSolidBot } from "react-icons/bi";
import { IoClose, IoSend } from "react-icons/io5";
import { BsArrowsFullscreen } from "react-icons/bs";
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

// ==== Gợi ý phù hợp Tư tưởng HCM ====
const suggestions: SuggestionItemProps[] = [
  {
    title: "Tư tưởng Hồ Chí Minh về độc lập dân tộc gắn với CNXH là gì?",
    icon: "🇻🇳",
  },
  {
    title: "Đạo đức cách mạng theo Hồ Chí Minh gồm những điểm chính nào?",
    icon: "🌿",
  },
  { title: "Đại đoàn kết dân tộc trong Tư tưởng Hồ Chí Minh", icon: "🤝" },
  { title: "Nhà nước của dân, do dân, vì dân được hiểu thế nào?", icon: "🏛️" },
];

const STORAGE_KEY = "hcm_chat_messages_v1";
const LEGACY_KEYS = [
  "hcm_chat_popup_messages_v1",
  "hcm_chat_fullpage_messages_v1",
];
const MAX_HISTORY = 20;

export default function ChatPopup() {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const messageEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [messages, setMessages] = useState<MessageItemProps[]>([]);

  // ==== helper: load/save history ====
  type MessageItemProps = {
    content: string;
    sender: "me" | "model";
    timestamp?: Date;
  };

  // hợp nhất/di trú từ các khóa cũ sang khóa mới
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
  // Init messages from storage (or default greeting)
  useEffect(() => {
    const restored = migrateAndLoadHistory();
    if (restored && restored.length) {
      setMessages(restored);
    } else {
      setMessages([
        {
          content:
            "Xin chào! 👋 Mình là **Gia sư Tư tưởng Hồ Chí Minh**. Bạn có thể hỏi về khái niệm, luận điểm cốt lõi (độc lập dân tộc gắn CNXH, dân chủ, đạo đức cách mạng, đại đoàn kết, nhà nước của dân–do dân–vì dân, giáo dục–văn hóa, xây dựng Đảng), hoặc các câu chuyện/tiểu sử về Bác.",
          sender: "model",
          timestamp: new Date(),
        },
      ]);
    }
  }, []);

  // Save history whenever messages change
  useEffect(() => {
    if (messages.length) saveHistory(messages);
  }, [messages]);

  // Auto-scroll (chỉ khi đang ở gần đáy)
  useEffect(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    const threshold = 120; // px từ đáy
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
    } catch (error) {
      console.error(error);
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

  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(suggestion);
  };

  return (
    <>
      {/* Floating Button */}
      <div
        className={`fixed bottom-6 right-6 w-16 h-16 group hover:scale-110 bg-gradient-to-br from-yellow-400 to-yellow-500 rounded-full flex justify-center items-center cursor-pointer z-60 duration-300 transition-all shadow-lg hover:shadow-xl ${
          !isOpen
            ? "opacity-100 translate-y-0"
            : "opacity-0 translate-y-20 pointer-events-none"
        }`}
        onClick={() => setIsOpen(true)}
      >
        <BiSolidBot className="text-black text-3xl group-hover:text-4xl duration-300 transition-all group-hover:rotate-12" />
        <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full animate-pulse"></div>
      </div>

      {/* Chat Window */}
      <div
        className={`fixed bottom-6 right-6 w-[400px] h-[600px] bg-gradient-to-b from-zinc-900 to-zinc-950 rounded-2xl flex flex-col shadow-2xl border border-zinc-800 duration-300 transition-all overflow-hidden ${
          isOpen
            ? "opacity-100 translate-x-0 scale-100"
            : "opacity-0 translate-x-full scale-95 pointer-events-none"
        } min-h-0`}
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-yellow-500 to-yellow-600 px-5 py-4 flex justify-between items-center flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
              <BiSolidBot className="text-yellow-600 text-2xl" />
            </div>
            <div>
              <div className="font-bold text-white text-sm">
                Gia sư Tư tưởng Hồ Chí Minh
              </div>
              <div className="text-xs text-yellow-100 flex items-center gap-1">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                Đang hoạt động
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              className="p-2 hover:bg-white/20 rounded-lg transition-all"
              onClick={() =>
                window.open(
                  `${window.location.origin}/chat`,
                  "_blank",
                  "noopener,noreferrer"
                )
              }
              title="Mở trang chat đầy đủ"
            >
              <BsArrowsFullscreen className="text-white text-lg" />
            </button>
            <button
              className="p-2 hover:bg-white/20 rounded-lg transition-all"
              onClick={() => setIsOpen(false)}
            >
              <IoClose className="text-white text-2xl" />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div
          ref={messagesContainerRef}
          className="chat-scroll flex-1 overflow-y-auto px-4 py-4 space-y-4 bg-zinc-950 no-scrollbar"
          style={{
            scrollBehavior: "smooth",
            overscrollBehavior: "contain" as const,
          }}
          data-lenis-prevent-wheel
          data-lenis-prevent-touch
          onWheel={(e) => {
            e.preventDefault(); // chặn Lenis
            e.stopPropagation(); // không nổi bọt ra window
            const el = messagesContainerRef.current;
            if (!el) return;
            el.scrollTop += e.deltaY; // tự cuộn container
          }}
          onWheelCapture={(e) => e.stopPropagation()}
          onTouchMoveCapture={(e) => e.stopPropagation()}
        >
          {/* Suggestions khi chỉ có lời chào */}
          {messages.length === 1 && messages[0].sender === "model" && (
            <div className="grid grid-cols-2 gap-2 mb-4">
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  onClick={() => handleSuggestionClick(s.title)}
                  className="p-3 bg-zinc-800 hover:bg-zinc-700 rounded-xl text-left text-sm text-zinc-300 hover:text-yellow-400 transition-all border border-zinc-700 hover:border-yellow-500/50"
                >
                  <div className="text-2xl mb-1">{s.icon}</div>
                  <div className="text-xs">{s.title}</div>
                </button>
              ))}
            </div>
          )}

          {messages.map((m, i) => (
            <MessageItem key={i} data={m} />
          ))}

          {isLoading && <TypingIndicator />}
          <div ref={messageEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 bg-zinc-900 border-t border-zinc-800 flex-shrink-0">
          <div className="flex gap-2 items-end">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              className="flex-1 bg-zinc-800 text-zinc-200 rounded-xl px-4 py-3 resize-none outline-none focus:ring-2 focus:ring-yellow-500 text-sm max-h-32"
              placeholder="Nhập câu hỏi của bạn..."
              rows={1}
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
              className="bg-yellow-500 hover:bg-yellow-600 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-all"
            >
              <IoSend className="text-xl" />
            </button>
          </div>
          <div className="text-xs text-zinc-500 mt-2 text-center">
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
        .overflow-y-auto::-webkit-scrollbar { width: 6px; }
        .overflow-y-auto::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); border-radius: 10px; }
        .overflow-y-auto::-webkit-scrollbar-thumb { background: rgba(234,179,8,0.5); border-radius: 10px; }
        .overflow-y-auto::-webkit-scrollbar-thumb:hover { background: rgba(234,179,8,0.7); }
      `}</style>
    </>
  );
}

// ===== Message Item
const MessageItem = ({ data }: { data: MessageItemProps }) => {
  const normalized = data.content
    .replaceAll("Sources:", "**Nguồn:**")
    .replaceAll("GT", "Giáo trình Tư tưởng Hồ Chí Minh")
    .replaceAll("ToanTap", "Hồ Chí Minh Toàn tập")
    .replaceAll("BupSenXanh", "Búp Sen Xanh (Sơn Tùng)");
  const parsedContent = marked(normalized);

  return (
    <div
      className={`flex gap-3 ${
        data.sender === "me" ? "justify-end" : "justify-start"
      }`}
    >
      {data.sender === "model" && (
        <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
          <BiSolidBot className="text-white text-lg" />
        </div>
      )}
      <div
        className={`max-w-[75%] px-4 py-3 rounded-2xl ${
          data.sender === "me"
            ? "bg-yellow-600 text-white rounded-tr-sm"
            : "bg-zinc-800 text-zinc-200 rounded-tl-sm"
        }`}
        style={{
          wordWrap: "break-word",
          overflowWrap: "break-word",
          wordBreak: "break-word",
        }}
      >
        <div
          className="text-sm message-content"
          dangerouslySetInnerHTML={{ __html: parsedContent }}
        />
        {data.timestamp && (
          <div className="text-xs opacity-50 mt-1">
            {new Date(data.timestamp).toLocaleTimeString("vi-VN", {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        )}
      </div>
      {data.sender === "me" && (
        <div className="w-8 h-8 bg-zinc-700 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
          <FaUser className="text-white text-sm" />
        </div>
      )}
    </div>
  );
};

// ===== Typing Indicator
const TypingIndicator = () => (
  <div className="flex gap-3 justify-start">
    <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center flex-shrink-0">
      <BiSolidBot className="text-white text-lg" />
    </div>
    <div className="bg-zinc-800 px-6 py-4 rounded-2xl rounded-tl-sm">
      <div className="flex gap-1.5">
        <div
          className="w-2 h-2 bg-zinc-600 rounded-full animate-bounce"
          style={{ animationDelay: "0ms" }}
        />
        <div
          className="w-2 h-2 bg-zinc-600 rounded-full animate-bounce"
          style={{ animationDelay: "150ms" }}
        />
        <div
          className="w-2 h-2 bg-zinc-600 rounded-full animate-bounce"
          style={{ animationDelay: "300ms" }}
        />
      </div>
    </div>
  </div>
);

// ===== API
const getAnswer = async (question: string) => {
  const body = { question, model_name: "gpt" };
  const response = await fetch(`https://hcm.jangkuz.io.vn`, {
    body: JSON.stringify(body),
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  return (await response.json())?.answer;
};
