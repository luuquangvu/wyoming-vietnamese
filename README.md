# Wyoming Vietnamese

Giải pháp nhận diện giọng nói (**STT** - Speech-to-Text) và phát giọng đọc (**TTS** - Text-to-Speech) tiếng Việt hoạt động hoàn toàn **cục bộ (local)**, riêng tư và tự nhiên dành riêng cho trợ lý ảo **Home Assistant (Assist)** thông qua giao thức chuẩn [Wyoming Protocol](https://github.com/OHF-Voice/wyoming).

Toàn bộ quá trình xử lý âm thanh và suy luận AI (inference) đều diễn ra ngay trên phần cứng của bạn trong mạng nội bộ (LAN): **không phụ thuộc dịch vụ đám mây (cloud-free), không mất phí API định kỳ và không gửi dữ liệu ra bên ngoài**.

> [!NOTE]
> Hệ thống hoạt động hoàn toàn ngoại tuyến (**100% offline**) sau khi đã tải các mô hình AI và giọng đọc về bộ nhớ đệm (cache) ở lần khởi động đầu tiên. Nếu bạn thay đổi danh sách giọng đọc (`TTS_VOICE`) hoặc chuyển sang engine khác chưa có sẵn trong volume lưu trữ, container sẽ cần kết nối Internet để nạp thêm các tệp tương ứng.

---

## Điểm nổi bật

- **Tất cả trong một (All-in-One: STT & TTS)**: Tích hợp đồng thời cả nhận diện giọng nói lẫn chuyển văn bản thành giọng đọc tiếng Việt trong duy nhất một container Docker, dùng chung một cổng mạng (`10300`), giúp tiết kiệm tài nguyên và tinh giản cấu hình trên Home Assistant.
- **Bảo mật và Riêng tư tuyệt đối (100% Local & Privacy)**: Mọi dữ liệu thu âm, câu lệnh điều khiển và phản hồi nhà thông minh của gia đình bạn đều được xử lý nội bộ, không bao giờ bị chuyển ra ngoài Internet.
- **Linh hoạt lựa chọn 2 engine TTS thế hệ mới**:
  - **Engine NghiTTS (`TTS_ENGINE: nghitts`)**: Xây dựng trên kiến trúc VITS (22.05 kHz) chạy qua runtime C++ `sherpa-onnx` tối ưu cao. Tốc độ phản hồi gần như tức thì, tiêu thụ rất ít tài nguyên phần cứng, rất lý tưởng cho Raspberry Pi 4/5, NAS hoặc các dòng Mini PC tiết kiệm điện.
  - **Engine ZeroTTS (`TTS_ENGINE: zerotts`)**: Sử dụng mô hình AI ngôn ngữ giọng nói ZeroTTS (định dạng GGUF Q8_0) kết hợp MOSS Audio Codec 48 kHz qua runtime C++ GGML. Chất âm chuẩn phòng thu, ngữ điệu truyền cảm và biểu cảm sống động như người thật.
- **Thư viện 28 giọng đọc phong phú**: Cung cấp sẵn 20 giọng NghiTTS và 8 giọng ZeroTTS với đầy đủ các vùng miền Bắc - Trung - Nam, giọng nam, giọng nữ, đa dạng phong cách từ trợ lý ảo, phát thanh viên, MC cho đến đọc truyện, tâm sự.
- **Xử lý ngắt nghỉ tự nhiên theo ngữ pháp**: Thuật toán tự động nhận diện cấu trúc câu (dấu chấm, phẩy, hai chấm...) và các đoạn văn để căn chỉnh khoảng lặng hợp lý, giúp câu thoại liền mạch, lưu loát, không bị dồn chữ hay cảm giác "đọc như máy".
- **Tự động cấu hình & Sẵn sàng chạy offline**: Tự động tải và kiểm tra toàn vẹn (checksum SHA-256) các mô hình ở lần khởi chạy đầu, lưu vào Docker volume và sẵn sàng vận hành lâu dài mà không cần duy trì kết nối Internet.

---

## So sánh nhanh 2 engine TTS

| Tiêu chí                      | Engine NghiTTS (`nghitts` - Mặc định)                             | Engine ZeroTTS (`zerotts`)                                                           |
| :---------------------------- | :---------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| **Kiến trúc cốt lõi**         | VITS qua runtime C++ `sherpa-onnx`                                | Neural Transformer (GGUF Q8_0) + MOSS Codec qua runtime C++ GGML                     |
| **Chất lượng âm thanh**       | 22.05 kHz (rõ ràng, mạch lạc, dễ nghe)                            | 48 kHz (chất lượng âm thanh phòng thu, chi tiết cao)                                 |
| **Tốc độ phản hồi (Latency)** | Siêu nhanh (< 0.2 giây), phản hồi gần như tức thì                 | Mượt mà (khoảng 0.5 đến 1.5 giây tùy hiệu năng CPU)                                  |
| **Đặc trưng giọng đọc**       | Rõ ràng, dứt khoát, chuẩn phong cách trợ lý ảo và phát thanh viên | Rất tự nhiên, giàu cảm xúc, nhấn nhá và ngữ điệu chân thực như người thật            |
| **Yêu cầu phần cứng**         | Rất nhẹ (phù hợp Raspberry Pi 4/5, Mini PC, NAS)                  | Yêu cầu CPU tương đối (khuyên dùng Mini PC x86 như Intel N100, Core i, AMD Ryzen...) |
| **Số lượng giọng đọc**        | 20 giọng (Bắc, Trung, Nam)                                        | 8 giọng (Bắc)                                                                        |

---

## Yêu cầu chuẩn bị

- **Phần cứng**:
  - Thiết bị chạy Docker: Raspberry Pi 4/5 (khuyên dùng với engine `nghitts`), x86 Mini PC (Intel N100, Core i3/i5/i7, AMD Ryzen...), NAS (Synology, QNAP, TrueNAS...) hoặc bất kỳ máy chủ Linux nào.
  - Bộ nhớ RAM: Khuyến nghị tối thiểu 1 GB RAM trống (với engine `nghitts`) hoặc 2 GB RAM trống (với engine `zerotts`).
  - Dung lượng ổ cứng: Trống khoảng 2 - 3 GB để lưu trữ mô hình nhận diện giọng nói và các giọng đọc TTS.
- **Môi trường & Mạng**:
  - Thiết bị đã cài đặt sẵn **Docker** và **Docker Compose**.
  - Kết nối mạng nội bộ (LAN): Thiết bị chạy Wyoming và máy chủ Home Assistant kết nối chung mạng nội bộ và có thể liên lạc thông suốt với nhau.
  - Cổng mạng: Cổng TCP `10300` khả dụng (chưa bị dịch vụ khác chiếm dụng).
  - Có kết nối Internet ở lần khởi động đầu tiên để hệ thống tải mô hình.

---

## Cài đặt nhanh bằng Docker Compose

Đây là phương thức tiện lợi và được khuyến nghị nhất cho cộng đồng người dùng Home Assistant. Tệp Compose sử dụng ảnh Docker dựng sẵn (pre-built image), bạn không cần tải mã nguồn hay tự biên dịch.

### 1. Tải file cấu hình và khởi chạy

Tạo thư mục làm việc và tải file cấu hình `docker-compose.online.yaml`:

```bash
mkdir -p ~/wyoming-vietnamese && cd ~/wyoming-vietnamese
curl -LO https://raw.githubusercontent.com/luuquangvu/wyoming-vietnamese/main/docker-compose.online.yaml
docker compose -f docker-compose.online.yaml up -d --pull always
```

### 2. Theo dõi quá trình nạp mô hình

Ở lần chạy đầu tiên, container sẽ tự động tải mô hình nhận diện giọng nói (STT Zipformer) và các giọng đọc TTS được chọn (thời gian tải thường từ 1 đến 3 phút tùy theo tốc độ mạng):

```bash
docker logs -f wyoming-vietnamese
```

Khi màn hình log xuất hiện thông báo:

```text
Wyoming STT/TTS service is ready at tcp://0.0.0.0:10300
```

Dịch vụ đã nạp xong toàn bộ mô hình và sẵn sàng để kết nối với Home Assistant!

---

## Kết nối với Home Assistant

Sau khi container đã khởi động thành công và mở cổng `10300`, bạn tiến hành kết nối theo các bước sau:

### Bước 1: Thêm tích hợp Wyoming Protocol

1. Trong giao diện Home Assistant, vào mục **Cài đặt (Settings)** > **Thiết bị & Dịch vụ (Devices & Services)**.
2. Bấm vào nút **Thêm tích hợp (Add Integration)** ở góc dưới bên phải.
3. Tìm kiếm **Wyoming Protocol** và chọn nó.
4. Nhập thông tin kết nối:
   - **Host**: Địa chỉ IP mạng LAN của máy chủ đang chạy Docker (ví dụ: `192.168.1.100`).
     _(Lưu ý: Không dùng `localhost` hay `127.0.0.1` nếu Home Assistant và Docker nằm trên hai thiết bị hoặc máy ảo khác nhau)._
   - **Port**: `10300` (hoặc cổng ngoài host bạn đã ánh xạ trong Docker Compose).
5. Bấm **Gửi (Submit)**. Home Assistant sẽ ngay lập tức nhận diện cả 2 dịch vụ STT và TTS tiếng Việt.

### Bước 2: Thiết lập Pipeline trợ lý giọng nói (Assist Pipeline)

1. Vào mục **Cài đặt (Settings)** > **Trợ lý giọng nói (Voice Assistants)**.
2. Chọn trợ lý bạn đang dùng (hoặc bấm **Thêm trợ lý** để tạo một pipeline tiếng Việt mới):
   - **Ngôn ngữ (Language)**: Chọn `Tiếng Việt (Vietnamese)`.
   - **Chuyển lời nói thành văn bản (Speech-to-text)**: Chọn `wyoming_vietnamese`.
   - **Chuyển văn bản thành lời nói (Text-to-speech)**: Chọn `wyoming_vietnamese`, sau đó chọn giọng đọc yêu thích trong danh sách thả xuống.
3. Bấm **Lưu (Save)**.

### Bước 3: Trải nghiệm thực tế

- Nhấp vào biểu tượng **Assist** (góc trên cùng bên phải giao diện Home Assistant).
- Bấm vào biểu tượng micro hoặc gõ lệnh để thử nghiệm:
  - _"Mấy giờ rồi?"_
  - _"Bật đèn phòng khách"_
  - _"Nhiệt độ phòng ngủ hiện tại là bao nhiêu?"_
- Trợ lý Assist sẽ nhận diện chính xác từng câu lệnh tiếng Việt và phản hồi lại bằng giọng đọc truyền cảm!

---

## Danh sách giọng đọc và tùy biến TTS

Container hỗ trợ 2 bộ phát giọng đọc (TTS Engine) thông qua biến môi trường `TTS_ENGINE`:

1. **`nghitts` (Mặc định)**: Sử dụng các giọng đọc **NghiTTS** (mô hình VITS 22.05 kHz) qua runtime C++ `sherpa-onnx`. Tốc độ phản hồi siêu nhanh, chiếm ít tài nguyên, tối ưu hoàn hảo cho Raspberry Pi và Mini PC.
2. **`zerotts`**: Sử dụng các giọng đọc **ZeroTTS** (mô hình AI ngôn ngữ giọng nói định dạng GGUF Q8_0 kết hợp MOSS Codec 48 kHz qua runtime C++ GGML). Chất âm 48 kHz chuẩn phòng thu, ngữ điệu tự nhiên và biểu cảm vượt trội.

Bạn có thể chỉnh sửa tệp `docker-compose.online.yaml`, thay đổi `TTS_ENGINE` và khai báo các giọng muốn dùng trong `TTS_VOICE` (các mã giọng phân cách bằng dấu phẩy hoặc khoảng trắng):

- Mã giọng đứng **đầu tiên** sẽ luôn là **giọng đọc mặc định**.
- Các mã giọng tiếp theo trong danh sách sẽ được nạp sẵn và hiển thị trong danh mục lựa chọn của Home Assistant.

> [!IMPORTANT]
> Mỗi engine chỉ nhận danh sách mã giọng tương ứng của nó. Không cấu hình lẫn lộn mã giọng của `nghitts` sang `zerotts` hoặc ngược lại.

### Ví dụ 1: Cấu hình engine NghiTTS (Mặc định - Phản hồi siêu nhanh)

```yaml
environment:
  WYOMING_PORT: 10300
  TTS_ENGINE: "nghitts"
  TTS_VOICE: "ngoc-huyen-moi, duy-onyx-moi, thanh-phuong-viettel, ngoc-ngan, mai-phuong"
  LOG_LEVEL: "info"
```

### Ví dụ 2: Cấu hình engine ZeroTTS (Chất lượng cao - Biểu cảm tự nhiên)

```yaml
environment:
  WYOMING_PORT: 10300
  TTS_ENGINE: "zerotts"
  TTS_VOICE: "maichi, baotrang, giahuy, hamy, huuduc"
  LOG_LEVEL: "info"
```

> [!TIP]
> Biến môi trường chỉ có hiệu lực khi container được tạo mới. Sau khi chỉnh sửa `TTS_ENGINE` hoặc `TTS_VOICE`, bạn hãy chạy lệnh sau để Docker áp dụng ngay cấu hình mới:
>
> ```bash
> docker compose -f docker-compose.online.yaml up -d --pull always --force-recreate
> ```

### Bảng mã giọng đọc engine NghiTTS (VITS 22.05 kHz)

| Mã giọng (`id`)        | Tên hiển thị         | Vùng miền / Đặc trưng phong cách                                                                 |
| :--------------------- | :------------------- | :----------------------------------------------------------------------------------------------- |
| `ngoc-huyen-moi`       | Ngọc Huyền (mới)     | Nữ miền Bắc (trong trẻo, tự nhiên, thích hợp làm trợ lý hàng ngày, đọc review và tin tức)        |
| `ban-mai`              | Ban Mai              | Nữ miền Bắc (dịu dàng, chuẩn phát thanh viên, rất truyền cảm)                                    |
| `thanh-phuong-viettel` | Thanh Phương Viettel | Nữ miền Bắc (rõ ràng, lưu loát, dứt khoát, chuẩn phong cách tổng đài và trợ lý ảo chuyên nghiệp) |
| `mai-phuong`           | Mai Phương           | Nữ miền Bắc (nhẹ nhàng, ấm áp, thích hợp đọc sách nói và tin tức dài)                            |
| `phuong-trang`         | Phương Trang         | Nữ miền Bắc (trầm ấm, truyền cảm, phong cách thuyết minh)                                        |
| `duy-onyx-moi`         | Duy Onyx (mới)       | Nam miền Bắc (trầm ấm, hiện đại, ngữ điệu tự nhiên, rất hợp làm giọng trợ lý nam)                |
| `duy-oryx`             | Duy Oryx             | Nam miền Bắc (trầm, đĩnh đạc, chững chạc)                                                        |
| `minh-khang`           | Minh Khang           | Nam miền Bắc (trẻ trung, năng động, cuốn hút, phong cách kênh Kiến Giải Mã)                      |
| `minh-quang`           | Minh Quang           | Nam miền Bắc (chững chạc, phát âm chuẩn, phong cách bản tin thời sự)                             |
| `manh-dung`            | Mạnh Dũng            | Nam miền Bắc (hào sảng, khỏe khoắn, dứt khoát, phong cách phóng sự - ký sự)                      |
| `chieu-thanh`          | Chiếu Thành          | Nam miền Nam (chất giọng trầm ấm, phong cách đọc truyện kiếm hiệp và dã sử)                      |
| `thien-tam`            | Thiện Tâm            | Nam miền Nam (từ tốn, sâu lắng, điềm tĩnh, phong cách audio tâm sự và triết lý)                  |
| `ngoc-ngan`            | Ngọc Ngạn            | Nam miền Bắc (trầm, hóm hỉnh, phong cách MC kể chuyện Paris By Night đặc trưng)                  |
| `tran-thanh`           | Trấn Thành           | Nam miền Nam (hoạt ngôn, biểu cảm đa dạng, sinh động và vui vẻ)                                  |
| `viet-thao`            | Việt Thảo            | Nam miền Nam (hóm hỉnh, hoạt náo, gần gũi, phong cách MC sân khấu)                               |
| `tai-an`               | Tài An               | Nam miền Bắc (rành mạch, phong cách thuyết minh lịch sử CD Media)                                |
| `lac-phi`              | Lạc Phi              | Nữ miền Bắc (truyền cảm, phong cách thuyết minh và review phim)                                  |
| `my-tam`               | Mỹ Tâm               | Nữ miền Trung / Nam (chất giọng ấm áp đặc trưng của ca sĩ Mỹ Tâm)                                |
| `my-tam-real`          | Mỹ Tâm Real          | Nữ miền Trung / Nam (chất giọng đặc trưng của ca sĩ Mỹ Tâm, tự nhiên và chân thực)               |
| `adam`                 | adam                 | Nam quốc tế (âm sắc ElevenLabs Adam đọc tiếng Việt chuẩn xác)                                    |

### Bảng mã giọng đọc engine ZeroTTS (Neural 48 kHz)

| Mã giọng (`id`) | Tên hiển thị | Vùng miền / Đặc trưng phong cách                                                          |
| :-------------- | :----------- | :---------------------------------------------------------------------------------------- |
| `maichi`        | Mai Chi      | Nữ miền Bắc (nhẹ nhàng, thân thiện, tự nhiên như trò chuyện ngoài đời thực)               |
| `baotrang`      | Bảo Trang    | Nữ miền Bắc (phong thái trưởng thành, đĩnh đạc, rõ ràng, trung tính, rất hợp đọc tin tức) |
| `giahuy`        | Gia Huy      | Nam miền Bắc (giọng trẻ, trầm ấm, tâm tình, thích hợp kể chuyện và đối thoại thân mật)    |
| `hamy`          | Hà My        | Nữ miền Bắc (giọng trẻ, tông cao trong sáng, biểu cảm sinh động, phong cách năng động)    |
| `huuduc`        | Hữu Đức      | Nam miền Bắc (chất giọng lớn tuổi, điềm đạm, trầm ấm, phong cách kể chuyện truyền thống)  |
| `kimoanh`       | Kim Oanh     | Nữ miền Bắc (độ tuổi trung niên, ấm áp, giàu cảm xúc, phong cách đọc truyện và tâm tình)  |
| `quangminh`     | Quang Minh   | Nam miền Bắc (giọng trẻ, dứt khoát, sáng rõ, chuẩn phong cách phát thanh viên tin tức)    |
| `tiendat`       | Tiến Đạt     | Nam miền Bắc (giọng trẻ, sôi nổi, năng lượng cao, phong cách bình luận viên)              |

---

## Cấu hình chi tiết và Tùy chọn nâng cao

### Các biến môi trường thường dùng trong Compose

| Biến môi trường |           Mặc định            | Mô tả chi tiết                                                                                                                   |
| :-------------- | :---------------------------: | :------------------------------------------------------------------------------------------------------------------------------- |
| `WYOMING_PORT`  |            `10300`            | Cổng TCP dịch vụ lắng nghe cho giao thức Wyoming.                                                                                |
| `TTS_ENGINE`    |           `nghitts`           | Bộ tạo giọng đọc: `nghitts` (nhẹ, nhanh qua sherpa-onnx) hoặc `zerotts` (AI Neural 48 kHz qua GGML C-FFI).                       |
| `TTS_VOICE`     | _(giọng mặc định của engine)_ | Danh sách mã giọng kích hoạt, phân cách bằng dấu phẩy hoặc khoảng trắng (giọng đầu tiên là mặc định).                            |
| `LOG_LEVEL`     |            `info`             | Mức độ chi tiết của nhật ký hệ thống (`debug`, `info`, `warning`, `error`).                                                      |
| `TZ`            |      _(Chưa thiết lập)_       | Múi giờ hệ thống để log hiển thị đúng giờ địa phương (ví dụ: `Asia/Ho_Chi_Minh`; chỉ có hiệu lực khi được truyền vào container). |

### Tinh chỉnh nâng cao (Dành cho người dùng chuyên sâu)

Khi cần tối ưu hiệu năng hoặc kiểm soát chi tiết hơn, bạn có thể bổ sung các biến sau vào mục `environment` trong tệp Compose:

| Biến môi trường            | Mặc định | Mô tả chi tiết                                                                                                                                                                                     |
| :------------------------- | :------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CPU_THREADS`              |   `0`    | Số luồng CPU sử dụng cho việc suy luận (`0` là tự động nhận diện và tận dụng toàn bộ số luồng CPU khả dụng).                                                                                       |
| `OFFLINE`                  | `false`  | Khi đặt thành `true`, container sẽ chặn mọi yêu cầu tải mô hình qua Internet và chỉ sử dụng các mô hình đã có sẵn trong volume.                                                                    |
| `TTS_PARAGRAPH_SILENCE_MS` |  `600`   | Khoảng lặng tối thiểu giữa các đoạn văn hoặc khi xuống dòng (đơn vị: mili-giây, chỉ áp dụng cho engine `nghitts`).                                                                                 |
| `TTS_SENTENCE_SILENCE_MS`  |  `400`   | Khoảng lặng tối thiểu giữa các câu kết thúc bằng dấu `.`, `!`, `?` (đơn vị: mili-giây, chỉ áp dụng cho engine `nghitts`). Bạn có thể tăng giá trị này nếu muốn giọng đọc khoan thai, chậm rãi hơn. |
| `TTS_CLAUSE_SILENCE_MS`    |  `200`   | Khoảng lặng tối thiểu sau các dấu ngắt vế câu `,`, `;`, `:` (đơn vị: mili-giây, chỉ áp dụng cho engine `nghitts`).                                                                                 |
| `MAX_STT_AUDIO_SECONDS`    | `120.0`  | Thời lượng âm thanh tối đa cho một lượt nhận diện giọng nói STT (giây).                                                                                                                            |
| `MAX_TTS_TEXT_CHARS`       |  `2000`  | Giới hạn độ dài tối đa của văn bản gửi đến TTS trong một yêu cầu (ký tự).                                                                                                                          |

> [!IMPORTANT]
> Hai Docker volume `cache` và `models` đóng vai trò lưu trữ toàn bộ các mô hình AI và dữ liệu giọng đọc đã tải về. Bạn **không nên xóa** hai volume này để container có thể khởi động lại tức thì và hoạt động hoàn toàn ngoại tuyến (offline).

---

## Các phương án triển khai khác

### 1. Cài đặt dưới dạng Home Assistant Add-on

Nếu bạn đang sử dụng **Home Assistant OS (HAOS)** hoặc **Home Assistant Supervised** và muốn cài đặt trực tiếp dạng Add-on chỉ bằng vài cú click chuột từ giao diện của Home Assistant, vui lòng tham khảo kho Add-on: [luuquangvu/ha-addons](https://github.com/luuquangvu/ha-addons).

### 2. Chạy nhanh bằng lệnh `docker run`

Nếu không muốn tạo file Docker Compose, bạn có thể khởi chạy trực tiếp bằng một dòng lệnh Docker:

```bash
docker run -d \
  --name wyoming-vietnamese \
  --restart unless-stopped \
  -p 10300:10300 \
  -e TTS_ENGINE="nghitts" \
  -e TTS_VOICE="ngoc-huyen-moi, duy-onyx-moi, thanh-phuong-viettel, ngoc-ngan, mai-phuong" \
  -v wyoming-vietnamese-cache:/app/.cache \
  -v wyoming-vietnamese-models:/app/models \
  ghcr.io/luuquangvu/wyoming-vietnamese:latest
```

Khi muốn cập nhật ảnh mới hoặc thay đổi biến môi trường, hãy tải ảnh mới nhất, xóa container cũ rồi khởi chạy lại. Hai volume có tên (`wyoming-vietnamese-cache` và `wyoming-vietnamese-models`) vẫn được giữ nguyên nên mô hình không cần phải tải lại:

```bash
docker pull ghcr.io/luuquangvu/wyoming-vietnamese:latest
docker rm -f wyoming-vietnamese
```

### 3. Tự biên dịch ảnh Docker từ mã nguồn

Dành cho các lập trình viên hoặc người dùng muốn tùy biến sâu mã nguồn:

```bash
git clone https://github.com/luuquangvu/wyoming-vietnamese.git
cd wyoming-vietnamese
docker compose up --build -d
```

---

## Xử lý sự cố thường gặp (Troubleshooting)

### 1. Home Assistant báo lỗi không kết nối được tới Wyoming Protocol ("Failed to connect")

- **Kiểm tra trạng thái container**: Chạy lệnh `docker ps` xem container `wyoming-vietnamese` có đang trong trạng thái `Up` (đang chạy) hay không.
- **Xem log container**: Chạy lệnh `docker logs wyoming-vietnamese` để kiểm tra xem dịch vụ đã in dòng thông báo sẵn sàng ở cổng `10300` chưa.
- **Kiểm tra tường lửa (Firewall)**: Đảm bảo cổng `10300` trên máy chủ Docker không bị chặn bởi tường lửa hệ điều hành (như UFW, iptables trên Linux, hoặc Windows Firewall).
- **Kiểm tra địa chỉ IP**: Nhập chính xác địa chỉ IP trong mạng LAN của máy chủ Docker (ví dụ `192.168.1.100`), không sử dụng `localhost` nếu Home Assistant và Docker nằm trên hai thiết bị hoặc máy ảo khác nhau.

### 2. Container khởi động chậm hoặc bị thoát (exit) ở lần chạy đầu tiên

- **Kiểm tra kết nối Internet**: Ở lần chạy đầu tiên, container bắt buộc phải có Internet để tải mô hình nhận diện giọng nói (STT Zipformer) và các giọng đọc TTS. Hãy kiểm tra kết nối mạng của máy chủ Docker.
- **Xem tiến trình tải**: Mở log thời gian thực bằng lệnh `docker logs -f wyoming-vietnamese` để theo dõi tiến độ tải file và xác thực mã băm SHA-256.
- **Cấu hình phần cứng hạn chế**: Nếu thiết bị có dung lượng RAM thấp (dưới 2 GB), hãy chọn `TTS_ENGINE: "nghitts"` và chỉ cấu hình từ 1 đến 2 giọng đọc cần thiết nhất để tối ưu hóa bộ nhớ.

### 3. Đã đổi giọng trong `TTS_VOICE` nhưng Home Assistant không hiển thị giọng mới

- Biến môi trường trong Docker Compose chỉ được áp dụng khi container được tái tạo (recreate). Sau khi chỉnh sửa file cấu hình, bạn chạy lệnh:

  ```bash
  docker compose -f docker-compose.online.yaml up -d --pull always --force-recreate
  ```

- Sau khi container khởi động lại xong, hãy vào Home Assistant > **Cài đặt (Settings)** > **Thiết bị & Dịch vụ (Devices & Services)** > tìm tích hợp **Wyoming Protocol** > bấm vào biểu tượng dấu 3 chấm góc phải và chọn **Tải lại (Reload)** để Home Assistant đồng bộ danh sách giọng mới.

### 4. Giọng đọc bị giật cục hoặc phản hồi chậm trên Raspberry Pi

- Hãy chuyển sang sử dụng engine `nghitts` (`TTS_ENGINE: "nghitts"`). Engine này sử dụng mô hình VITS siêu nhẹ, được tối ưu hóa riêng cho các kiến trúc ARM như Raspberry Pi 4/5. Engine `zerotts` sử dụng mô hình ngôn ngữ giọng nói AI lớn hơn nhiều, chỉ phù hợp khi chạy trên các máy chủ có CPU x86 tương đối mạnh.

---

## Đóng góp và Hỗ trợ

- Báo lỗi hoặc đề xuất tính năng mới qua [GitHub Issues](https://github.com/luuquangvu/wyoming-vietnamese/issues). Vui lòng đính kèm log liên quan (và lưu ý che đi các thông tin nhạy cảm nếu có).
- Mọi đóng góp cải tiến mã nguồn thông qua Pull Requests đều rất được hoan nghênh!

---

## Lời cảm ơn

Dự án được xây dựng và hoàn thiện dựa trên các công trình mã nguồn mở xuất sắc:

- [nghimestudio/nghitts](https://github.com/nghimestudio/nghitts): Cung cấp các mô hình giọng đọc tiếng Việt (TTS) chất lượng cao cho engine `nghitts`.
- [zeroweight-ai/ZeroTTS](https://github.com/zeroweight-ai/ZeroTTS): Cung cấp mô hình ngôn ngữ giọng nói ZeroTTS và runtime C++ GGML cho engine `zerotts`.
- [hynt](https://huggingface.co/hynt): Cung cấp mô hình nhận diện giọng nói tiếng Việt `Zipformer-30M-RNNT-6000h` (STT).
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): Thư viện suy luận offline tối ưu cao cho cả STT và TTS.
- [Wyoming Protocol](https://github.com/OHF-Voice/wyoming): Chuẩn giao thức mở cho trợ lý giọng nói trong hệ sinh thái Home Assistant.

---

## Giấy phép

Dự án được phát hành dưới giấy phép mã nguồn mở **MIT License**. Xem chi tiết tại tệp [LICENSE](LICENSE).
