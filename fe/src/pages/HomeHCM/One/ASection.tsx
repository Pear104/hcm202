import React from "react";

export default function ASection() {
  return (
    <>
      <div className="p-[2vw]">
        <div className="grid grid-cols-2 gap-[6vw] mt-[4vw]">
          <div>
            <img
              className="w-full aspect-[10/9] object-cover object-center rounded-xl"
              src="images/1_A.png"
              alt=""
            />
          </div>
          <div className="">
            <div className="unbounded text-[4vw] text-[#FF2F2F] text-end capitalize leading-[4.4vw] pr-[1.5vw] border-r-[1vw]">
              A
            </div>
            <div className="unbounded text-[4vw] text-red-500/90 text-end capitalize">
              Kết hợp sức mạnh dân tộc với sức mạnh thời đại
            </div>
            <div className="inter italic text-end mt-[2vw]">
              Thực hiện đoàn kết quốc tế nhằm tập hợp lực lượng bên ngoài, tranh
              thủ sự đồng tình, ủng hộ và giúp đỡ của bạn bè quốc tế, đồng thời
              kết hợp sức mạnh dân tộc với sức mạnh của các trào lưu cách mạng
              thời đại để tạo thành sức mạnh tổng hợp cho cách mạng Việt Nam.
              Qua đó, cách mạng Việt Nam không chỉ giành thắng lợi cho dân tộc
              mình mà còn góp phần cùng nhân dân thế giới thực hiện thắng lợi
              các mục tiêu cách mạng chung của thời đại.
            </div>
          </div>
        </div>
        <div className="mt-[4vw] flex flex-col items-center">
          <div className="unbounded text-[3.6vw] text-red-500 text-center capitalize">
            Sức mạnh dân tộc và sức mạnh thời đại
          </div>
          <div className="w-[64vw] space-y-[1vw] mt-[2vw]">
            <div className="rounded-3xl px-[2vw] py-[1vw] border border-zinc-500">
              <div className="text-[1.5vw] unbounded uppercase">
                Sức mạnh dân tộc
              </div>
              <div className="h-0 overflow-hidden">
                Sức mạnh dân tộc là sự tổng hợp của các yếu tố vật chất và tinh
                thần, trước hết là sức mạnh của chủ nghĩa yêu nước, ý thức tự
                lực tự cường, tinh thần đoàn kết và ý chí đấu tranh anh dũng,
                bất khuất vì độc lập, tự do. Chính những yếu tố này đã giúp dân
                tộc Việt Nam vượt qua mọi khó khăn, thử thách trong sự nghiệp
                dựng nước và giữ nước.
              </div>
            </div>
            <div className="bg-red-500 rounded-3xl px-[2vw] py-[1vw]">
              <div className="text-[1.5vw] unbounded uppercase">
                Sức mạnh thời đại
              </div>
              <div className="h-fit overflow-hidden">
                Sức mạnh thời đại là sức mạnh của phong trào cách mạng thế giới,
                được hun đúc từ thành quả lý luận và thực tiễn của chủ nghĩa Mác
                – Lênin, đặc biệt được khẳng định qua thắng lợi vĩ đại của Cách
                mạng Tháng Mười Nga năm 1917. Đây còn là sức mạnh đến từ phong
                trào giải phóng dân tộc, phong trào cách mạng của giai cấp công
                nhân quốc tế, sự tiến bộ vượt bậc của khoa học – kỹ thuật và sự
                đồng tình, ủng hộ to lớn của nhân dân tiến bộ trên thế giới. Hồ
                Chí Minh đã sớm xác định cách mạng Việt Nam là một bộ phận khăng
                khít của cách mạng thế giới và chỉ có thể đi đến thành công khi
                gắn bó, đoàn kết chặt chẽ với phong trào cách mạng quốc tế. Đây
                là nhận thức mới mẻ, tiến bộ so với các bậc tiền bối, thể hiện
                rõ vai trò quyết định của đoàn kết quốc tế đối với thắng lợi của
                cách mạng Việt Nam.
              </div>
            </div>
          </div>
          <div className="unbounded text-[3.6vw] text-red-500 text-center capitalize py-[2vw]">
            Đại đoàn kết dân tộc gắn liền với
            <br />
            đoàn kết quốc tế
          </div>
          <div className="w-[64vw] text-[1.5vw] text-center">
            Theo Hồ Chí Minh, đại đoàn kết toàn dân tộc phải gắn liền với đoàn
            kết quốc tế. Đại đoàn kết dân tộc chính là cơ sở, là tiền đề vững
            chắc để triển khai thành công đoàn kết quốc tế. Chỉ khi sức mạnh dân
            tộc kết hợp hài hòa với sức mạnh thời đại thì mới có thể tạo nên sức
            mạnh tổng hợp, bảo đảm cho thắng lợi của cách mạng Việt Nam.
          </div>
        </div>
      </div>
    </>
  );
}
