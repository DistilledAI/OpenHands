# PlanningAgent

PlanningAgent là một agent chuyên về lập kế hoạch và thực hiện các tác vụ phức tạp theo phương pháp từng bước.

## Tổng quan

PlanningAgent mở rộng từ GeneralAgent và bổ sung thêm khả năng lập kế hoạch. Các chức năng chính bao gồm:

1. Tạo kế hoạch ban đầu dựa trên yêu cầu người dùng
2. Theo dõi trạng thái thực hiện của từng bước trong kế hoạch
3. Thực hiện từng bước một cách tuần tự
4. Cung cấp báo cáo tổng kết sau khi hoàn thành kế hoạch

Agent sử dụng PlanningTool để lưu trữ và quản lý kế hoạch.

## Cách sử dụng

### Khởi tạo

```python
from openhands.agenthub.planning_agent import PlanningAgent
from openhands.core.config import AgentConfig
from openhands.llm.llm import LLM

# Khởi tạo LLM với mô hình phù hợp
llm = LLM(model="gpt-4")

# Khởi tạo cấu hình agent
config = AgentConfig()

# Khởi tạo PlanningAgent
planning_agent = PlanningAgent(llm=llm, config=config)
```

### Chạy agent

PlanningAgent cung cấp phương thức `run()` để xử lý toàn bộ quy trình lập kế hoạch và thực hiện:

```python
# Đầu vào là yêu cầu từ người dùng
input_text = "Tạo một ứng dụng web đơn giản hiển thị thời tiết hiện tại"

# Chạy agent và theo dõi tiến trình
async for response in planning_agent.run(input_text):
    print(f"Type: {response['mtype']}")
    print(f"Content: {response['content']}")
    print("-" * 50)
```

## Luồng hoạt động

1. **Tạo kế hoạch ban đầu**: Agent sẽ phân tích yêu cầu và tạo một kế hoạch với các bước cụ thể.
2. **Thực hiện từng bước**: Agent thực hiện tuần tự từng bước trong kế hoạch, sử dụng các công cụ thích hợp.
3. **Theo dõi tiến độ**: Trạng thái của mỗi bước được cập nhật liên tục (chưa bắt đầu, đang tiến hành, đã hoàn thành, bị chặn).
4. **Tổng kết**: Sau khi hoàn thành tất cả các bước, agent tạo một bản tóm tắt kết quả.

## Các loại phản hồi

PlanningAgent trả về các loại phản hồi khác nhau trong quá trình thực thi:

- `planning`: Thông tin về kế hoạch hiện tại và trạng thái của các bước.
- `error`: Thông báo lỗi nếu có vấn đề xảy ra.
- `final_answer`: Bản tóm tắt kết quả cuối cùng sau khi hoàn thành kế hoạch.

## Công cụ lập kế hoạch

Agent sử dụng PlanningTool để quản lý kế hoạch, bao gồm các chức năng:

- Tạo kế hoạch mới với các bước cụ thể
- Cập nhật trạng thái của từng bước
- Thêm kết quả cho mỗi bước đã hoàn thành
- Theo dõi tiến độ tổng thể của kế hoạch
