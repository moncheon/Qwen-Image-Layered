# Qwen-Image-Layered 프로젝트 가이드

## 프로젝트 개요
이미지를 여러 RGBA 레이어로 분해하는 AI 모델. 각 레이어를 독립적으로 편집 가능.

---

## 핵심 Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Qwen-Image-Layered Flow                       │
└─────────────────────────────────────────────────────────────────────┘

1. 환경 초기화
   ├── print_environment_info()     # Python, PyTorch, CUDA, 라이브러리 버전 확인
   ├── clear_memory()               # gc.collect() + torch.cuda.empty_cache()
   └── ENV_INFO                     # 사용 가능한 백엔드 정보 저장

2. Pipeline 로딩 (load_pipeline)
   ├── setup_cuda_optimizations()   # TF32, cuDNN benchmark, SDPA 설정
   ├── get_model_path()             # 로컬 모델 확인 또는 HuggingFace에서 다운로드
   ├── QwenImageLayeredPipeline.from_pretrained()
   │   └── [옵션] quantization_config  # 4bit/8bit 양자화 (현재 비활성화)
   │
   ├── CPU Offload 설정 (CPU_OFFLOAD_MODE에 따라)
   │   ├── "off"        → pipe.to("cuda")
   │   ├── "model"      → pipe.enable_model_cpu_offload()
   │   └── "sequential" → pipe.enable_sequential_cpu_offload() ★현재 설정
   │
   ├── VAE 최적화
   │   ├── pipe.vae.enable_slicing()
   │   └── pipe.vae.enable_tiling()
   │
   └── FirstBlockCache 적용 (threshold=0.1)

3. 추론 (infer 함수)
   ├── 입력 이미지 전처리 (→ RGBA 변환)
   ├── Generator 생성 (CPU Offload 모드에 따라 device 결정)
   │
   ├── with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
   │       output = pipeline(**inputs)
   │
   ├── clear_memory()               # 추론 후 VRAM/RAM 정리
   │
   └── 출력 생성
       ├── PNG 레이어 이미지들
       ├── PPTX 파일
       ├── ZIP 파일
       └── PSD 파일

4. Gradio 웹 UI
   └── demo.launch(server_name="0.0.0.0", server_port=7869)
```

---

## 최적화 분류

### VRAM 최적화

| 최적화 | 적용 여부 | 설명 |
|--------|----------|------|
| **Sequential CPU Offload** | O (핵심) | VRAM 부족 시 레이어 단위로 RAM에 offload |
| **Model CPU Offload** | 옵션 | 모델 단위로 RAM에 offload |
| **VAE Slicing** | O | VAE 연산을 슬라이스로 분할 |
| **VAE Tiling** | O | 큰 이미지를 타일로 분할 처리 |
| **bfloat16** | O | FP32 대비 메모리 50% 절약 |
| **Memory Efficient SDPA** | O | 메모리 효율적 Attention |
| **4bit 양자화** | X | 잡음만 나옴 (품질 저하 심각) |
| **8bit 양자화** | X | VRAM 24GB 초과 가능성 |

### 성능/기타 최적화

| 최적화 | 적용 여부 | 설명 |
|--------|----------|------|
| **TF32** | O | Ada/Ampere GPU 연산 가속 |
| **cuDNN Benchmark** | O | 최적 알고리즘 자동 선택 |
| **FirstBlockCache** | O | 추론 속도 20-40% 향상 |
| **torch.inference_mode** | O | 그래디언트 계산 비활성화 |
| **torch.amp.autocast** | O | Mixed Precision 추론 |
| **gc.collect + empty_cache** | O | 명시적 메모리 정리 |
| **Flash Attention** | X | Windows에서 불안정 |
| **SageAttention** | X | PyTorch 2.4 호환성 문제 |
| **torch.compile** | X | Segmentation Fault 발생 |
| **xformers** | X | dual-stream 구조와 호환 안됨 |

---

## 설정 플래그 (app.py 상단)

```python
ENABLE_SAGE_ATTENTION = False      # SageAttention (PyTorch 2.4 호환성 문제)
QUANTIZATION_MODE = None           # None, "4bit", "4bit_quality", "8bit"
ENABLE_FIRST_BLOCK_CACHE = True    # FirstBlockCache 활성화
FIRST_BLOCK_CACHE_THRESHOLD = 0.1  # FirstBlockCache threshold
CPU_OFFLOAD_MODE = "sequential"    # "off", "model", "sequential"
```

### CPU_OFFLOAD_MODE 상세

| 모드 | VRAM 사용량 | 속도 | 설명 |
|------|------------|------|------|
| `"off"` | 높음 | 빠름 | GPU만 사용, VRAM 넘치면 임의로 RAM 사용 |
| `"model"` | 중간 | 중간 | 모델 단위로 GPU↔RAM 이동 |
| `"sequential"` | 최소 | 느림 | 레이어 단위로 GPU↔RAM 이동 (RTX 4090 24GB에서 권장) |

---

## 파일 구조

```
Qwen-Image-Layered/
├── src/
│   ├── app.py                 # 메인 Gradio 앱
│   └── tool/
│       ├── combine_layers.py  # 레이어 합성 도구
│       └── edit_rgba_image.py # RGBA 이미지 편집 도구
├── models/
│   └── Qwen-Image-Layered/    # 로컬 모델 캐시
├── assets/
│   └── test_images/           # 예제 이미지
├── .venv/                     # Python 가상환경
├── README.md                  # 프로젝트 설명
├── CLAUDE.md                  # 이 파일
└── todo/
    └── optimize4090.md        # 최적화 작업 기록
```

---

## 실행 방법

```bash
# 가상환경 활성화
.venv\Scripts\activate

# 메인 앱 실행
python src/app.py

# 레이어 편집 도구
python src/tool/edit_rgba_image.py

# 레이어 합성 도구
python src/tool/combine_layers.py
```

---

## 주요 의존성

- `transformers>=4.51.3`
- `diffusers` (git+https://github.com/huggingface/diffusers)
- `torch` (CUDA 12.1)
- `gradio`
- `accelerate`
- `python-pptx`
- `psd-tools`

---

## 핵심 최적화 원리: Sequential CPU Offload

```
┌─────────────────────────────────────────────────────────────────────┐
│  Sequential CPU Offload (VRAM 부족 시 레이어 단위 RAM Offload)       │
└─────────────────────────────────────────────────────────────────────┘

[GPU VRAM 24GB]                    [System RAM]
┌──────────────┐                   ┌──────────────┐
│  Layer 1     │ ◄──── 실행 중 ────│              │
│  (active)    │                   │  Layer 2     │
│              │                   │  Layer 3     │
│              │                   │  Layer 4     │
│              │                   │  ...         │
└──────────────┘                   └──────────────┘

1. 필요한 레이어만 GPU에 로드
2. 연산 완료 후 RAM으로 이동
3. 다음 레이어를 GPU로 로드
4. 반복

장점: VRAM 24GB로도 대형 모델 실행 가능
단점: GPU↔RAM 데이터 전송으로 속도 저하
```