# Wyoming Vietnamese

Giải pháp nhận diện giọng nói (STT) và phát giọng đọc (TTS) tiếng Việt **cục bộ (local)**, riêng tư và tự nhiên cho trợ lý **Home Assistant (Assist)** qua giao thức [Wyoming Protocol](https://github.com/OHF-Voice/wyoming).

Toàn bộ quá trình xử lý giọng nói và suy luận diễn ra ngay trên thiết bị trong mạng nội bộ (LAN) của bạn: không cần tài khoản đám mây, không tốn phí API và không gửi dữ liệu ra bên ngoài (lưu ý: dịch vụ hoạt động ngoại tuyến sau khi tất cả mô hình và giọng đang sử dụng đã được tải xuống; nếu đổi `TTS_VOICE` hoặc thêm mô hình chưa có trong volume, cần kết nối Internet để tải chúng).

---

## Điểm nổi bật

- **2 trong 1 (STT + TTS)**: Tích hợp cả nhận diện giọng nói (Speech-to-Text) lẫn phát giọng đọc (Text-to-Speech) trong một container duy nhất, dùng chung một cổng mạng (`10300`).
- **Cục bộ và Riêng tư**: Dữ liệu giọng nói và văn bản của gia đình bạn không bao giờ rời khỏi mạng nội bộ (quá trình suy luận chạy 100% offline sau khi đã nạp mô hình ở lần khởi động đầu).
- **Tốc độ cao và Tiết kiệm tài nguyên**: Sử dụng mô hình Zipformer tiếng Việt (STT) và VITS NghiTTS chạy qua bộ suy luận C++ tối ưu `sherpa-onnx`, phản hồi nhanh ngay cả trên các thiết bị có cấu hình vừa và yếu (Raspberry Pi 4/5, Mini PC, NAS).
- **20 giọng đọc tiếng Việt phong phú**: Đầy đủ giọng miền Bắc, miền Nam, nam và nữ với ngữ điệu tự nhiên, rõ ràng.
- **Ngắt nghỉ tự nhiên theo chuẩn ngữ pháp**: Tự động căn chỉnh khoảng nghỉ giữa đoạn văn, câu và vế câu (dấu phẩy), giúp câu văn mạch lạc, không bị dồn chữ.
- **Tự động tải và Sẵn sàng chạy ngoại tuyến**: Tự động tải mô hình ở lần khởi động đầu tiên, lưu vào Docker volume và có thể hoạt động hoàn toàn không cần Internet sau khi các mô hình/giọng cần dùng đã được tải.

---

## Yêu cầu chuẩn bị

- Một thiết bị cài đặt **Docker** và **Docker Compose** (Raspberry Pi 4/5, x86 Mini PC, máy chủ Linux, NAS Synology/QNAP...).
- Home Assistant và thiết bị chạy container có thể kết nối với nhau qua mạng nội bộ.
- Khoảng **2 GB dung lượng trống** để lưu trữ mô hình STT và các giọng đọc TTS.
- Kết nối Internet ở lần chạy đầu tiên để nạp mô hình.
- Cổng TCP `10300` chưa bị chiếm dụng (nếu đổi cổng host ở mục `ports` trong Docker Compose, ví dụ `"10303:10300"`, bạn cần nhập cổng mới này vào ô **Port** khi kết nối với Home Assistant).

---

## Cài đặt nhanh bằng Docker Compose

Đây là phương thức khuyến nghị và tiện lợi nhất cho phần lớn người dùng. Tệp Compose sử dụng ảnh Docker dựng sẵn (pre-built image), bạn không cần tải mã nguồn hay tự biên dịch.

### 1. Tải tệp cấu hình và khởi chạy

Chạy lệnh sau trên máy chủ Docker của bạn:

```bash
curl -LO https://raw.githubusercontent.com/luuquangvu/wyoming-vietnamese/main/docker-compose.online.yaml
docker compose -f docker-compose.online.yaml up -d
```

### 2. Theo dõi quá trình nạp mô hình

Ở lần chạy đầu tiên, container sẽ tự động tải mô hình nhận diện giọng nói và các giọng TTS được cấu hình (thường mất 1-3 phút tùy tốc độ mạng):

```bash
docker logs -f wyoming-vietnamese
```

Khi nhật ký hiển thị thông báo sẵn sàng (ví dụ):

```text
Wyoming STT/TTS service is ready at tcp://0.0.0.0:10300
```

(hoặc dòng nhật ký chứa `Wyoming STT/TTS service is ready at ...`), dịch vụ đã sẵn sàng để kết nối với Home Assistant!

---

## Kết nối với Home Assistant

Sau khi container đã khởi động thành công:

1. Trong Home Assistant, vào **Cài đặt (Settings) > Thiết bị & Dịch vụ (Devices & Services)**.
2. Nhấn nút **Thêm tích hợp (Add Integration)** ở góc dưới bên phải.
3. Tìm kiếm **Wyoming Protocol** và chọn nó.
4. Điền thông tin kết nối:
   - **Host**: Địa chỉ IP của máy đang chạy Docker (ví dụ: `192.168.1.100`; nếu Home Assistant chạy chung trên cùng một máy, bạn có thể nhập IP nội bộ của máy đó).
   - **Port**: `10300` (hoặc cổng host đã ánh xạ ở mục `ports` trong Docker Compose).
5. Nhấn **Gửi (Submit)**. Home Assistant sẽ tự động nhận diện cả 2 dịch vụ STT và TTS tiếng Việt.
6. Vào **Cài đặt (Settings) > Trợ lý giọng nói (Voice Assistants)**, chọn trợ lý bạn đang dùng (Assist pipeline):
   - **Chuyển lời nói thành văn bản (Speech-to-text)**: Chọn `wyoming-vietnamese`.
   - **Chuyển văn bản thành lời nói (Text-to-speech)**: Chọn `wyoming-vietnamese` và chọn giọng đọc ưa thích.
7. Nhấn vào biểu tượng **Assist** ở góc trên cùng bên phải giao diện Home Assistant để thử ra lệnh (ví dụ: _"Mấy giờ rồi?"_, _"Bật đèn phòng khách"_).

---

## Danh sách giọng đọc và tùy chỉnh giọng

Mở tệp `docker-compose.online.yaml` và chỉnh sửa biến `TTS_VOICE`. Bạn có thể khai báo một hoặc nhiều mã giọng, ngăn cách bằng dấu phẩy hoặc khoảng trắng:

- Mã giọng đứng **đầu tiên** sẽ là giọng đọc mặc định.
- Các giọng còn lại sẽ xuất hiện trong danh sách lựa chọn của Home Assistant.

```yaml
environment:
  WYOMING_PORT: 10300
  TTS_VOICE: "ngoc-huyen-moi, duy-onyx-moi, thanh-phuong-viettel, ngoc-ngan, mai-phuong"
  LOG_LEVEL: "info"
```

> [!TIP]
> Biến môi trường chỉ được nạp khi container được tạo mới. Sau khi thay đổi `TTS_VOICE`, hãy chạy lệnh sau để áp dụng:
>
> ```bash
> docker compose -f docker-compose.online.yaml up -d --force-recreate
> ```

### Bảng mã giọng đọc có sẵn

| Mã giọng (`id`)        | Tên hiển thị         | Vùng miền / Đặc trưng                                                  | Mặc định |
| :--------------------- | :------------------- | :--------------------------------------------------------------------- | :------: |
| `ngoc-huyen-moi`       | Ngọc Huyền (mới)     | Nữ miền Bắc (tự nhiên, trong trẻo, phong cách đọc truyện và review)    |  **Có**  |
| `ban-mai`              | Ban Mai              | Nữ miền Bắc (dịu dàng, truyền cảm, phong cách phát thanh viên)         |          |
| `thanh-phuong-viettel` | Thanh Phương Viettel | Nữ miền Bắc (rõ ràng, lưu loát, chuẩn phong cách trợ lý và tổng đài)   |          |
| `mai-phuong`           | Mai Phương           | Nữ miền Bắc (nhẹ nhàng, ấm áp, phong cách đọc sách nói)                |          |
| `phuong-trang`         | Phương Trang         | Nữ miền Bắc (trầm ấm, truyền cảm, phong cách thuyết minh)              |          |
| `duy-onyx-moi`         | Duy Onyx (mới)       | Nam miền Bắc (trầm ấm, tự nhiên, phong cách trợ lý nam)                |          |
| `duy-oryx`             | Duy Oryx             | Nam miền Bắc (trầm, đĩnh đạc)                                          |          |
| `minh-khang`           | Minh Khang           | Nam miền Bắc (trẻ trung, cuốn hút, phong cách kênh Kiến Giải Mã)       |          |
| `minh-quang`           | Minh Quang           | Nam miền Bắc (chững chạc, rõ ràng, phong cách đọc tin tức)             |          |
| `manh-dung`            | Mạnh Dũng            | Nam miền Bắc (hào sảng, dứt khoát, phong cách ký sự và tài liệu)       |          |
| `chieu-thanh`          | Chiếu Thành          | Nam miền Nam (trầm ấm, phong cách kể chuyện kiếm hiệp và dã sử)        |          |
| `thien-tam`            | Thiện Tâm            | Nam miền Nam (từ tốn, sâu lắng, phong cách tâm sự và audio Phật giáo)  |          |
| `ngoc-ngan`            | Ngọc Ngạn            | Nam miền Bắc (trầm, hóm hỉnh, phong cách MC dẫn chuyện Paris By Night) |          |
| `tran-thanh`           | Trấn Thành           | Nam miền Nam (hoạt ngôn, biểu cảm, phong cách nghệ sĩ hài hước)        |          |
| `viet-thao`            | Việt Thảo            | Nam miền Nam (hóm hỉnh, gần gũi, phong cách MC sân khấu)               |          |
| `tai-an`               | Tài An               | Nam miền Bắc (rành mạch, phong cách thuyết minh lịch sử CD Media)      |          |
| `lac-phi`              | Lạc Phi              | Nữ miền Bắc (truyền cảm, phong cách thuyết minh và review phim)        |          |
| `my-tam`               | Mỹ Tâm               | Nữ miền Nam / Miền Trung (giọng ca sĩ Mỹ Tâm, âm vị chuẩn toàn quốc)   |          |
| `my-tam-real`          | Mỹ Tâm Real          | Nữ miền Nam (giọng ca sĩ Mỹ Tâm, ngữ điệu miền Nam chân thực)          |          |
| `adam`                 | adam                 | Nam quốc tế (chất giọng ElevenLabs Adam đọc tiếng Việt)                |          |

---

## Cấu hình chi tiết và tùy chọn nâng cao

### Các thiết lập mặc định trong Compose

- `WYOMING_PORT`: Cổng TCP dịch vụ lắng nghe (mặc định: `10300`).
- `TTS_VOICE`: Danh sách các giọng TTS được tải và kích hoạt (giọng đầu tiên là mặc định).
- `LOG_LEVEL`: Mức độ chi tiết của nhật ký (`info`, `debug`, `warning`, `error`).

### Tùy chọn nâng cao (Dành cho người dùng chuyên sâu)

Khi cần tối ưu hóa hoặc kiểm soát chi tiết hơn, bạn có thể thêm các biến môi trường sau vào phần `environment` của tệp Compose:

| Biến môi trường            |      Mặc định      | Ý nghĩa và Hướng dẫn                                                                                                            |
| :------------------------- | :----------------: | :------------------------------------------------------------------------------------------------------------------------------ |
| `TZ`                       | `Asia/Ho_Chi_Minh` | Múi giờ địa phương để hiển thị thời gian trong log chính xác.                                                                   |
| `CPU_THREADS`              |        `0`         | Số luồng CPU sử dụng cho suy luận (`0` là tự động dùng tất cả luồng khả dụng).                                                  |
| `OFFLINE`                  |      `false`       | Đặt `"true"` sau khi đã tải đủ mô hình để chỉ nạp mô hình từ bộ nhớ đệm cục bộ (báo lỗi nếu thiếu file thay vì kết nối tải về). |
| `TTS_PARAGRAPH_SILENCE_MS` |       `700`        | Khoảng lặng tối thiểu giữa các đoạn văn hoặc ngắt dòng (đơn vị: mili-giây).                                                     |
| `TTS_SENTENCE_SILENCE_MS`  |       `500`        | Khoảng lặng tối thiểu giữa các câu kết thúc bằng dấu `.`, `!`, `?`. Tăng giá trị này nếu muốn giọng đọc chậm rãi hơn.           |
| `TTS_CLAUSE_SILENCE_MS`    |       `300`        | Khoảng lặng tối thiểu sau dấu phẩy `,`, chấm phẩy `;`, hai chấm `:` trong câu.                                                  |

> [!IMPORTANT]
> Hai volume `cache` và `models` lưu trữ toàn bộ mô hình đã tải về. Không nên xóa hai volume này để container khởi động tức thì ở các lần sau và có thể hoạt động ngoại tuyến.

---

## Các phương án triển khai khác

### 1. Cài đặt dưới dạng Home Assistant Add-on

Nếu bạn đang dùng **Home Assistant OS** (HAOS) hoặc **Supervised** và muốn cài đặt trực tiếp dạng Add-on từ giao diện Home Assistant, vui lòng tham khảo kho Add-on: [luuquangvu/ha-addons](https://github.com/luuquangvu/ha-addons).

### 2. Chạy nhanh bằng lệnh `docker run`

Nếu không muốn dùng Docker Compose:

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

Nếu thay đổi một biến `-e` hoặc cập nhật ảnh container, hãy xóa container cũ rồi chạy lại lệnh trên. Hai volume có tên (`wyoming-vietnamese-cache` và `wyoming-vietnamese-models`) vẫn được giữ nguyên, nên mô hình đã tải về không bị mất:

```bash
docker rm -f wyoming-vietnamese
```

### 3. Tự dựng ảnh Docker từ mã nguồn

Dành cho lập trình viên hoặc người muốn tùy biến mã nguồn:

```bash
git clone https://github.com/luuquangvu/wyoming-vietnamese.git
cd wyoming-vietnamese
docker compose up --build -d
```

---

## Xử lý sự cố thường gặp

### Home Assistant báo lỗi không kết nối được tới Wyoming Protocol

- Kiểm tra container có đang chạy không: `docker ps`.
- Xem nhật ký container: `docker logs wyoming-vietnamese`.
- Đảm bảo cổng `10300` không bị chặn bởi tường lửa (UFW, iptables, Windows Firewall...).
- Nhập chính xác địa chỉ IP của máy chủ Docker, tránh dùng `localhost` nếu Home Assistant và Docker nằm trên hai thiết bị khác nhau.

### Container khởi động chậm hoặc dừng đột ngột ở lần đầu

- Kiểm tra kết nối Internet của máy chủ Docker. Lần đầu cần tải mô hình nhận diện giọng nói (~150MB) và các giọng đọc (~60MB mỗi giọng).
- Nếu bạn vô tình bật `OFFLINE: "true"` trước khi tải đủ mô hình, container sẽ báo lỗi. Đổi lại `OFFLINE: "false"`, chạy lại và đợi nạp xong.

### Đã đổi giọng trong TTS_VOICE nhưng Home Assistant không hiện giọng mới

- Biến môi trường chỉ được nạp lại khi tái tạo container. Hãy chạy:

  ```bash
  docker compose -f docker-compose.online.yaml up -d --force-recreate
  ```

- Sau đó, vào Home Assistant, mở tích hợp **Wyoming Protocol** và chọn **Tải lại (Reload)**.

---

## Đóng góp và Hỗ trợ

- Báo lỗi hoặc đề xuất tính năng mới qua [GitHub Issues](https://github.com/luuquangvu/wyoming-vietnamese/issues). Vui lòng đính kèm nhật ký liên quan (và che đi các thông tin nhạy cảm).
- Mọi đóng góp cải tiến mã nguồn (Pull Requests) đều được chào đón!

---

## Lời cảm ơn

Dự án được xây dựng dựa trên các công trình mã nguồn mở xuất sắc:

- [nghimestudio/nghitts](https://github.com/nghimestudio/nghitts): Cung cấp các mô hình giọng nói tiếng Việt (TTS) chất lượng cao.
- [hynt](https://huggingface.co/hynt): Cung cấp mô hình nhận diện giọng nói tiếng Việt `Zipformer-30M-RNNT-6000h` (STT).
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): Bộ máy suy luận offline tối ưu cao cho cả STT và TTS.
- [Wyoming Protocol](https://github.com/OHF-Voice/wyoming): Chuẩn giao tiếp giọng nói mở cho hệ sinh thái Home Assistant.

---

## Giấy phép

Dự án được phát hành dưới giấy phép mã nguồn mở **MIT License**. Xem chi tiết tại tệp [LICENSE](LICENSE).
