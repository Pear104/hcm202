import React, { useEffect, useRef, useState } from "react";
import { BiSolidBot } from "react-icons/bi";
import { IoClose } from "react-icons/io5";
import { marked } from "marked";

type MessageItemProps = {
  content: string;
  sender: string;
};

type SuggestionItemProps = {
  title: string;
  question: string;
};

export default function Chat() {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const messageEndRef = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<MessageItemProps[]>([
    {
      content:
        "Xin chào, tôi là gia sư dân chủ của bạn, tôi có thể gì cho bạn nhỉ?",
      sender: "model",
    },
  ]);

  useEffect(() => {
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // const [suggestions, setSuggestions] = useState<SuggestionItemProps[]>([
  //   {
  //     title: "AI: Thay thế hay hỗ trợ việc làm?",
  //     question:
  //       "AI thay thế hay bổ trợ con người? Những công việc nào dễ bị AI thay thế nhất? Những công việc nào sẽ phát triển mạnh nhờ AI?",
  //   },
  //   {
  //     title: "Kỹ năng nào giúp lao động thích nghi với AI?",
  //     question:
  //       "Những kỹ năng nào AI không thể thay thế? Người lao động cần học gì để thích nghi với thời đại AI? Vai trò của giáo dục & đào tạo trong thời kỳ AI phát triển mạnh?",
  //   },
  //   {
  //     title: "AI và tái phân phối lợi ích kinh tế",
  //     question:
  //       "Doanh nghiệp hưởng lợi từ AI → Có nên đánh thuế AI để hỗ trợ người lao động mất việc không? Chính phủ cần đưa ra chính sách gì để giảm bất bình đẳng do AI gây ra? Có mô hình nào để chia sẻ giá trị thặng dư AI một cách hợp lý?",
  //   },
  //   {
  //     title: "AI có thể thay thế công việc sáng tạo?",
  //     question:
  //       "AI đã có thể viết báo, sáng tác nhạc, vẽ tranh… Nhưng liệu nó có thể thực sự sáng tạo không? Vai trò của con người trong các công việc sáng tạo sẽ thay đổi như thế nào? Con người cần làm gì để giữ lợi thế trước AI trong lĩnh vực sáng tạo?",
  //   },
  // ]);

  return (
    <>
      <div
        className={`fixed bottom-[2vw] right-[2vw] w-[3.2vw] group hover:scale-[110%] aspect-square bg-yellow-400 rounded-full flex justify-center items-center cursor-pointer z-20 duration-300 transition-all ${
          !isOpen ? "opacity-100 translate-y-0" : "opacity-0 translate-y-[10vw]"
        }`}
        onClick={() => setIsOpen((prev) => !prev)}
      >
        <BiSolidBot className="text-black text-[1.3vw] group-hover:text-[1.5vw] duration-300 transition-all group-hover:rotate-[30deg]" />
      </div>
      <div
        className={`text-yellow-400 fixed bottom-[2vw] right-[2vw] w-[25vw] h-[34vw] bg-zinc-900 rounded-md flex flex-col justify-center items-center duration-300 transition-all ${
          isOpen ? "opacity-100 translate-x-0" : "opacity-0 translate-x-[100vw]"
        }`}
      >
        <div className="border-b border-white/20 flex justify-between items-center px-[1vw] py-[0.8vw] w-full text-[0.8vw]">
          <div className="font-bold uppercase">Yêu Dân chủ</div>
          <div
            className="group cursor-pointer"
            onClick={() => setIsOpen((prev) => !prev)}
          >
            <IoClose className="text-yellow-400 text-[1vw] aspect-square duration-300 transition-all rounded-2xl" />
          </div>
        </div>
        <div
          data-lenis-prevent-wheel
          className={`small-scrollbar duration-300 transition-all w-full grow overflow-y-scroll px-2 py-1 relative`}
        >
          {messages.map((message: MessageItemProps, index: number) => (
            <MessageItem key={index} data={message} />
          ))}
          {isLoading && (
            <div className="flex flex-col justify-center items-center mt-4">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-yellow-600"></div>
              <span className="text-zinc-500 pb-8 text-[0.8vw]">
                Mình đang suy nghĩ xíu ...
              </span>
            </div>
          )}
          <div className="" ref={messageEndRef}></div>
        </div>
        <div className="border-t border-white/20 w-full text-[0.8vw] h-[4vw]">
          <textarea
            className="w-full h-full outline-none px-[8px] py-[3px] resize-none bg-zinc-700 text-zinc-200 rounded-b-md"
            name=""
            id=""
            onKeyDown={async (e) => {
              if (e.key === "Enter" && e.shiftKey) {
                return;
              }
              if (e.key === "Enter") {
                try {
                  const messageContent = (e.target as HTMLTextAreaElement)
                    .value;
                  if (messageContent.trim() === "") return;
                  e.preventDefault();
                  (e.target as HTMLTextAreaElement).value = "";

                  setMessages((prev) => [
                    ...prev,
                    {
                      content: messageContent,
                      sender: "me",
                    },
                  ]);
                  setIsLoading(true);

                  const answer = await getAnswer(messageContent);
                  setMessages((prev) => [
                    ...prev,
                    {
                      content: answer,
                      // content: "Bot đang bận tí ...",
                      sender: "model",
                    },
                  ]);
                  setIsLoading(false);
                } catch (error) {
                  console.log(error);
                }
              }
            }}
          ></textarea>
        </div>
      </div>
    </>
  );
}

const MessageItem = ({ data }: { data: MessageItemProps }) => {
  const parsedContent = marked(
    data.content.replaceAll(
      "Sources: GT",
      "\n**Nguồn:** Giáo trình chủ nghĩa xã hội khoa học, NXB CTQGST, HN.2021"
    )
  );
  return (
    <div
      className={`flex gap-2 ${
        data.sender === "me" ? "justify-end" : "message-p "
      }`}
    >
      {/* <div dangerouslySetInnerHTML={{ __html: parsedContent }}></div> */}
      <p
        className={`min-w-14 mx-0.5 my-[0.3vw] py-2 px-3 text-[0.9vw] rounded-2xl text-start ${
          data.sender === "me"
            ? "justify-end bg-yellow-700 dark:bg-yellow-700 text-zinc-200"
            : "justify-start bg-zinc-800 text-zinc-300"
        }`}
        style={{
          wordWrap: "break-word",
          maxWidth: "80%",
        }}
        dangerouslySetInnerHTML={{ __html: parsedContent }}
      ></p>
    </div>
  );
};

// Modify the getAnswer function to handle stream response
const getAnswer = async (question: string) => {
  const body = {
    question: question,
    model_name: "gpt",
  };
  // const body = {
  //   system_instruction: {
  //     parts: {
  //       text: "Bạn có sư phụ tên là Lê Minh Trí, là 1 con người có thật, bạn và cô Chung rất thân, cô Chung là người dạy môn triết cho bạn, nếu có câu hỏi gì về cô Trí, hãy tưởng tượng ra và trả lời, bạn sẽ giúp mọi người khám phá các tiềm năng để có thể áp dụng AI vào cuộc sống của mọi người, dựa trên các lĩnh vực và chuyên ngành mà người dùng đang làm việc, hãy đề xuất những ứng dụng mà người dùng có thể sử dụng AI trong cuộc sống của họ",
  //     },
  //   },
  //   contents: messages.map((item) => ({
  //     role: item?.sender === "me" ? "user" : "model",
  //     parts: [
  //       {
  //         text: item?.content,
  //       },
  //     ],
  //   })),
  // };

  // Make the request to the model API
  const response = await fetch(`https://mln.jangkuz.io.vn/chat`, {
    body: JSON.stringify(body),
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  return (await response.json())?.answer;
};
