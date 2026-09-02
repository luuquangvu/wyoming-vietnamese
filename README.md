# Wyoming Vietnamese

Để Assist của Home Assistant nghe và trả lời bằng tiếng Việt tự nhiên, ngay trong mạng nhà bạn.

Wyoming Vietnamese cung cấp cả dịch vụ **chuyển giọng nói thành văn bản (STT)** và **chuyển văn bản thành giọng nói (TTS)** qua Wyoming Protocol. Home Assistant chỉ cần kết nối tới một địa chỉ và một cổng (port) duy nhất; sau đó bạn có thể chọn dịch vụ này làm chuỗi xử lý (pipeline) giọng nói cho Assist.

## Điểm nổi bật

- Nhận diện tiếng Việt và chuyển văn bản thành giọng nói ngay trong mạng nhà bạn.
- Không cần tài khoản đám mây hoặc khóa API (API key); âm thanh và văn bản không phải gửi tới dịch vụ bên ngoài trong quá trình sử dụng.
- Có 20 giọng TTS tiếng Việt để chọn. Giọng đầu tiên trong cấu hình là giọng mặc định.
- Chạy bằng một container Docker duy nhất và dùng chung cổng Wyoming `10300` cho cả STT lẫn TTS.
- Mô hình (model) được tải một lần rồi lưu trong phân vùng lưu trữ (volume) Docker, nên các lần khởi động sau nhanh hơn. Sau khi tải xong, dịch vụ có thể chạy hoàn toàn ngoại tuyến (offline).

## Trước khi bắt đầu

Bạn cần:

- Một máy chạy Docker và Docker Compose, nằm trên cùng mạng với Home Assistant.
- Kết nối Internet ở lần khởi động đầu tiên để tải mô hình STT và các giọng TTS.
- Một cổng mạng (TCP port) chưa được sử dụng. Mặc định dịch vụ dùng cổng `10300`.

Home Assistant không cần chạy Docker trên cùng máy với Wyoming Vietnamese. Nếu chạy container trên máy khác, bạn chỉ cần dùng địa chỉ IP của máy đó khi thêm tích hợp (integration) Wyoming trong Home Assistant.

## Cài đặt nhanh bằng Docker Compose

Đây là cách phù hợp với hầu hết người dùng. Tệp Compose (Compose file) dùng ảnh Docker dựng sẵn (pre-built Docker image), nên bạn không cần tải mã nguồn về hoặc tự dựng ảnh.

```bash
curl -LO https://raw.githubusercontent.com/luuquangvu/wyoming-vietnamese/main/docker-compose.online.yaml
docker compose -f docker-compose.online.yaml up -d
```

Xem nhật ký (log) khởi động bằng:

```bash
docker logs -f wyoming-vietnamese
```

Lần đầu container có thể mất vài phút vì phải tải mô hình. Hãy chờ tới khi container khởi động hoàn tất trước khi kết nối Home Assistant.

### Kết nối với Home Assistant

Trong Home Assistant:

1. Mở **Cài đặt → Thiết bị & dịch vụ**.
2. Chọn **Thêm tích hợp** và tìm **Wyoming Protocol**.
3. Nhập địa chỉ IP hoặc tên máy đang chạy container.
4. Nhập cổng `10300`, rồi hoàn tất việc thêm tích hợp.
5. Mở **Cài đặt → Trợ lý giọng nói**, chọn chuỗi xử lý Assist bạn đang dùng và đặt Wyoming Vietnamese làm dịch vụ STT và TTS.

Sau đó, thử một câu đơn giản trong Assist. Dịch vụ sẽ xuất hiện với cả khả năng nhận diện tiếng Việt và các giọng đọc TTS đã cấu hình.

## Chọn giọng đọc

Mở `docker-compose.online.yaml` và sửa `TTS_VOICE`. Bạn có thể đặt một hoặc nhiều mã giọng, ngăn cách bằng dấu phẩy hoặc khoảng trắng. Mã đầu tiên là giọng mặc định; các mã còn lại sẽ được Home Assistant hiển thị để lựa chọn.

Ví dụ:

```yaml
environment:
  TTS_VOICE: "ngoc-huyen-moi, duy-onyx-moi, thanh-phuong-viettel, ngoc-ngan, mai-phuong"
```

Các biến môi trường (environment variable) chỉ được đọc khi container được tạo. Vì vậy, sau khi sửa `TTS_VOICE` hoặc bất kỳ biến nào khác trong phần `environment`, hãy **tạo lại (recreate) container**. Chỉ khởi động lại (restart) bằng `docker restart` sẽ không áp dụng cấu hình mới.

Áp dụng cấu hình mới bằng:

```bash
docker compose -f docker-compose.online.yaml up -d --force-recreate
```

Các giọng có sẵn:

| Mã                     | Tên hiển thị         | Mặc định |
| ---------------------- | -------------------- | :------: |
| `ban-mai`              | Ban Mai              |          |
| `chieu-thanh`          | Chiếu Thành          |          |
| `duy-onyx-moi`         | Duy Onyx (mới)       |          |
| `duy-oryx`             | Duy Oryx             |          |
| `lac-phi`              | Lạc Phi              |          |
| `mai-phuong`           | Mai Phương           |          |
| `minh-khang`           | Minh Khang           |          |
| `minh-quang`           | Minh Quang           |          |
| `manh-dung`            | Mạnh Dũng            |          |
| `my-tam`               | Mỹ Tâm               |          |
| `my-tam-real`          | Mỹ Tâm Real          |          |
| `ngoc-huyen-moi`       | Ngọc Huyền (mới)     |    ✓     |
| `ngoc-ngan`            | Ngọc Ngạn            |          |
| `phuong-trang`         | Phương Trang         |          |
| `thanh-phuong-viettel` | Thanh Phương Viettel |          |
| `thien-tam`            | Thiện Tâm            |          |
| `tran-thanh`           | Trấn Thành           |          |
| `tai-an`               | Tài An               |          |
| `viet-thao`            | Việt Thảo            |          |
| `adam`                 | adam                 |          |

## Các cách chạy khác

### Cài đặt dưới dạng Home Assistant Add-on

Nếu bạn đang sử dụng Home Assistant OS hoặc Supervised và muốn cài đặt trực tiếp dưới dạng App (Add-on) thay vì chạy container Docker độc lập, vui lòng xem hướng dẫn chi tiết tại kho lưu trữ [luuquangvu/ha-addons](https://github.com/luuquangvu/ha-addons).

### Chạy trực tiếp bằng Docker

Nếu bạn không dùng Docker Compose, có thể chạy trực tiếp ảnh Docker:

```bash
docker run -d \
  --name wyoming-vietnamese \
  --restart unless-stopped \
  -p 10300:10300 \
  -e TTS_VOICE="ngoc-huyen-moi, duy-onyx-moi, thanh-phuong-viettel, ngoc-ngan, mai-phuong" \
  -v wyoming-vietnamese-cache:/app/.cache \
  -v wyoming-vietnamese-models:/app/models \
  ghcr.io/luuquangvu/wyoming-vietnamese:latest
```

Nếu thay đổi một biến `-e`, hãy xóa container cũ rồi chạy lại lệnh trên. Hai phân vùng lưu trữ có tên (named volume) vẫn được giữ nguyên, nên bạn không phải tải lại mô hình:

```bash
docker rm -f wyoming-vietnamese
```

### Tự dựng ảnh Docker (build image) từ mã nguồn

```bash
git clone https://github.com/luuquangvu/wyoming-vietnamese.git
cd wyoming-vietnamese
docker compose up --build -d
```

Tệp [`docker-compose.yaml`](docker-compose.yaml) chứa cùng nhóm cấu hình môi trường như ảnh dựng sẵn và phù hợp khi bạn muốn tự dựng phiên bản của riêng mình.

## Chạy ngoại tuyến sau lần đầu

Sau khi container đã khởi động thành công và tải đủ mô hình, đặt:

```yaml
environment:
  OFFLINE: "true"
```

Tạo lại container để áp dụng; chỉ khởi động lại container là chưa đủ. Ở chế độ này, dịch vụ chỉ sử dụng các tệp đã có trong phân vùng lưu trữ (volume); nếu thiếu mô hình, container sẽ báo lỗi thay vì cố kết nối Internet.

## Tùy chỉnh thường gặp

Các giá trị sau có sẵn trong tệp Compose:

- `CPU_THREADS`: số luồng CPU dùng cho suy luận; để `0` để tự động dùng số luồng phù hợp.
- `TTS_SENTENCE_SILENCE_MS`: khoảng nghỉ giữa các câu. Tăng giá trị này nếu giọng đọc hơi nhanh.
- `TTS_CLAUSE_SILENCE_MS`: khoảng nghỉ sau dấu phẩy và các dấu câu trong mệnh đề.
- `TTS_SILENCE_JITTER_PERCENT`: thêm một chút thay đổi ngẫu nhiên vào khoảng nghỉ để câu đọc tự nhiên hơn.
- `LOG_LEVEL`: đặt `debug` khi cần xem nhật ký chi tiết; thông thường nên giữ `info`.

Không nên xóa hai phân vùng lưu trữ (volume) `cache` và `models` nếu bạn muốn giữ mô hình đã tải. Docker Compose sẽ giữ chúng qua các lần cập nhật hoặc tạo lại container.

## Xử lý sự cố

> **Không thêm được Wyoming trong Home Assistant**

- Kiểm tra container đang chạy: `docker ps`.
- Xem nhật ký: `docker logs wyoming-vietnamese`.
- Đảm bảo Home Assistant truy cập được máy chạy Docker và cổng TCP `10300` không bị tường lửa (firewall) chặn.
- Nếu container chạy trên máy khác, hãy nhập IP của máy đó, không phải `localhost` của Home Assistant.

> **Container vẫn đang tải hoặc khởi động chưa xong**

Theo dõi nhật ký bằng `docker logs -f wyoming-vietnamese`. Lần đầu cần Internet và có thể lâu hơn những lần sau. Nếu bật `OFFLINE: "true"` trước khi tải đủ mô hình, hãy đổi lại thành `"false"`, tạo lại container, rồi chờ quá trình tải hoàn tất.

> **Đổi giọng nhưng Home Assistant vẫn đọc bằng giọng cũ**

Kiểm tra mã giọng trong bảng trên, tạo lại container để nạp biến `TTS_VOICE`, rồi mở lại trang Trợ lý giọng nói trong Home Assistant.

## Đóng góp

Nếu bạn gặp lỗi, hãy gửi một báo cáo lỗi kèm cấu hình và nhật ký liên quan. Nhớ xóa thông tin riêng tư trước khi đăng để cộng đồng dễ dàng hỗ trợ và cải thiện dự án.

## Ghi công

Dự án sử dụng các công trình mã nguồn mở sau:

- [nghimestudio/nghitts](https://github.com/nghimestudio/nghitts) cung cấp các mô hình giọng đọc tiếng Việt.
- [hynt](https://huggingface.co/hynt) cung cấp mô hình nhận diện `Zipformer-30M-RNNT-6000h`.
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) cung cấp bộ máy suy luận STT và TTS.
- [Wyoming Protocol](https://github.com/OHF-Voice/wyoming) giúp kết nối dịch vụ với hệ sinh thái Home Assistant Voice.

## Giấy phép

Dự án được phát hành dưới **Giấy phép MIT**. Xem tệp [LICENSE](LICENSE) để biết thêm thông tin chi tiết.
