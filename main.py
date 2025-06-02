import streamlit as st
import requests
import base64
import json
import time
import zipfile
import io
from PIL import Image
from datetime import datetime
from pathlib import Path
import random

# 페이지 설정
st.set_page_config(
    page_title="NovelAI Automation Tool Lite",
    page_icon="🎨",
    layout="wide"
)

# CSS 스타일링 (반응형 추가)
st.markdown("""
<style>
/* 기존 앵커 링크 비활성화 */
.element-container .stMarkdown h1 .anchor-link,
.element-container .stMarkdown h2 .anchor-link,
.element-container .stMarkdown h3 .anchor-link,
.element-container .stMarkdown h4 .anchor-link,
.element-container .stMarkdown h5 .anchor-link,
.element-container .stMarkdown h6 .anchor-link {
    display: none !important;
    pointer-events: none !important;
}            
.element-container .stMarkdown h1,
.element-container .stMarkdown h2,
.element-container .stMarkdown h3,
.element-container .stMarkdown h4,
.element-container .stMarkdown h5,
.element-container .stMarkdown h6 {
    pointer-events: none !important;
    cursor: default !important;
}

/* 기존 스타일 */
.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 1rem;
    color: #4ECDC4;
}
.parameter-section {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}

/* Generation 섹션을 식별하기 위한 클래스 */
.generation-container {
    /* 기본 스타일 */
}

.tips-container {
    /* 기본 스타일 */
}

/* 중간 크기 화면 (1200px 이하) - PC 창이 좁아진 경우 */
@media (max-width: 1200px) {
    /* Streamlit 컬럼을 세로 배치로 변경 */
    .stHorizontalBlock > div {
        flex-direction: column !important;
    }
    
    /* 모든 컬럼을 full width로 */
    .stHorizontalBlock > div > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    .section-header {
        font-size: 1.4rem;
    }
}

/* 태블릿 크기 (900px 이하) */
@media (max-width: 900px) {
    .section-header {
        font-size: 1.3rem;
    }
}

/* 모바일 레이아웃 (768px 이하) */
@media (max-width: 768px) {
    .section-header {
        font-size: 1.25rem;
    }
    
    /* 사이드바가 접힌 상태에서도 잘 보이도록 */
    .stSidebar {
        width: auto;
    }
}

/* 매우 작은 화면 (480px 이하) */
@media (max-width: 480px) {
    .section-header {
        font-size: 1.1rem;
    }
    
    .parameter-section {
        padding: 0.75rem;
    }
}

/* 매우 넓은 화면에서는 다시 2컬럼으로 */
@media (min-width: 1201px) {
    .stHorizontalBlock > div {
        flex-direction: row !important;
    }
}
</style>
""", unsafe_allow_html=True)

# NovelAI API 관련 상수
NAI_BASE_URL = "https://image.novelai.net"
NAI_GENERATE_ENDPOINT = "/ai/generate-image"

# 모델 매핑
MODEL_MAPPINGS = {
    "NAI Diffusion V3": {
        "generate": "nai-diffusion-3",
        "inpainting": "nai-diffusion-3-inpainting"
    },
    "NAI Diffusion V4.5": {
        "generate": "nai-diffusion-4-5-full",
        "inpainting": "nai-diffusion-4-5-full-inpainting"
    },
    "NAI Diffusion V4.5 Curated": {
        "generate": "nai-diffusion-4-curated-preview",
        "inpainting": "nai-diffusion-4-curated-inpainting"
    },
    "NAI Diffusion V4": {
        "generate": "nai-diffusion-4-full",
        "inpainting": "nai-diffusion-4-full-inpainting"
    },
    "NAI Diffusion V4 Curated": {
        "generate": "nai-diffusion-4-curated-preview",
        "inpainting": "nai-diffusion-4-curated-inpainting"
    },
    "NAI Diffusion Furry V3": {
        "generate": "nai-diffusion-furry-3",
        "inpainting": "nai-diffusion-furry-3-inpainting"
    }
}

SAMPLERS = [
    "k_euler",
    "k_euler_ancestral", 
    "k_dpmpp_sde",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_2m",
    "k_dpmpp_2m_sde"
]

def generate_nai_image(access_token, prompt, model, parameters):
    """NovelAI 이미지 생성 함수"""
    
    # 퀄리티 태그 추가
    if parameters.get('quality_toggle', True):
        if "nai-diffusion-3" in model:
            prompt += ', {best quality}, {amazing quality}'
        elif 'nai-diffusion-4' in model:
            if "text" not in prompt:
                prompt += ", no text, best quality, amazing quality, very aesthetic, absurdres"
            else:
                prompt += ", best quality, amazing quality, very aesthetic, absurdres"
    
    # 요청 데이터 구성
    data = {
        "input": prompt,
        "model": model,
        "action": "generate",
        "parameters": parameters,
    }
    
    # NAI V4 특화 설정
    if 'nai-diffusion-4' in model:
        data['parameters'].update({
            'params_version': 3,
            'add_original_image': True,
            'characterPrompts': [],
            'legacy': False,
            'legacy_uc': False,
            'autoSmea': parameters.get('sm', True),
            'legacy_v3_extend': False,
            'prefer_brownian': True,
            'ucPreset': 0,
            'use_coords': False,
            'v4_negative_prompt': {
                'caption': {
                    'base_caption': parameters.get('negative_prompt', ''),
                    'char_captions': []
                },
                'legacy_uc': False
            },
            'v4_prompt': {
                'caption': {
                    'base_caption': prompt,
                    'char_captions': []
                },
                'use_coords': False,
                'use_order': True
            }
        })
        
        # 불필요한 파라미터 제거
        for key in ['sm', 'sm_dyn', 'enable_hr', 'enable_AD']:
            data['parameters'].pop(key, None)
    
    # API 요청
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.post(
        f"{NAI_BASE_URL}{NAI_GENERATE_ENDPOINT}", 
        json=data, 
        headers=headers, 
        timeout=180
    )
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    return response.content

def process_nai_response(content):
    """NAI 응답 처리"""
    try:
        zipped = zipfile.ZipFile(io.BytesIO(content))
        file_info = zipped.infolist()[0]
        image_bytes = zipped.read(file_info)
        
        image = Image.open(io.BytesIO(image_bytes))
        
        return image
    except Exception as e:
        raise Exception(f"Failed to process response: {str(e)}")


# 메인 애플리케이션
def main():
    # 사이드바 - API 설정
    with st.sidebar:
        st.markdown('<h1 class="section-header"> NAI Automation Tool</h1>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">🔑 API Settings</h2>', unsafe_allow_html=True)
        
        access_token = st.text_input(
            "NovelAI Access Token",
            type="password",
            help="NovelAI 계정에서 발급받은 API 토큰을 입력하세요"
        )
        
        if not access_token:
            st.warning("⚠️ API 토큰을 입력해주세요")
            return
        
        st.markdown('<h2 class="section-header">🎯 Model Settings</h2>', unsafe_allow_html=True)
        
        selected_model = st.selectbox(
            "Model",
            list(MODEL_MAPPINGS.keys()),
            index=1,  # NAI Diffusion V4 기본 선택
            help="사용할 NovelAI 모델을 선택하세요"
        )
        
        model_name = MODEL_MAPPINGS[selected_model]["generate"]
    
    # 화면 크기 감지를 위한 JavaScript (선택사항)
    st.markdown("""
    <script>
    function detectScreenSize() {
        const width = window.innerWidth;
        if (width <= 768) {
            document.body.classList.add('mobile-layout');
        } else {
            document.body.classList.remove('mobile-layout');
        }
    }
    
    window.addEventListener('resize', detectScreenSize);
    detectScreenSize();
    </script>
    """, unsafe_allow_html=True)
    
    # 화면 크기에 따른 조건부 렌더링
    # JavaScript로 화면 크기 체크
    screen_width_js = """
    <script>
    function updateScreenWidth() {
        const width = window.innerWidth;
        const widthElement = document.getElementById('screen-width');
        if (widthElement) {
            widthElement.textContent = width;
        }
    }
    
    window.addEventListener('resize', updateScreenWidth);
    updateScreenWidth();
    </script>
    <div id="screen-width" style="display: none;"></div>
    """
    
    st.markdown(screen_width_js, unsafe_allow_html=True)
    
    # 미디어 쿼리 기반 모바일 체크용 히든 엘리먼트
    mobile_check = st.markdown("""
    <div id="mobile-detector" style="display: none;">
        <div class="desktop-only">desktop</div>
        <div class="mobile-only">mobile</div>
    </div>
    <style>
    .desktop-only { display: block; }
    .mobile-only { display: none; }
    
    @media (max-width: 768px) {
        .desktop-only { display: none; }
        .mobile-only { display: block; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 기본적으로는 Streamlit 컬럼 사용 (PC)
    col1, col2 = st.columns([1, 1])
    
    # 왼쪽 컬럼 - 프롬프트 입력 전용
    with col1:
        st.markdown('<h2 class="section-header">📝 Prompt Settings</h2>', unsafe_allow_html=True)
        
        # 프롬프트 입력
        prompt = st.text_area(
            "Prompt",
            height=200,
            placeholder="1girl, anime style, beautiful, detailed...",
            help="생성하고 싶은 이미지를 설명하는 프롬프트를 입력하세요",
            key="main_prompt"
        )
        
        negative_prompt = st.text_area(
            "Negative Prompt",
            height=150,
            placeholder="lowres, bad quality, blurry...",
            help="이미지에서 제외하고 싶은 요소들을 입력하세요",
            key="main_negative"
        )

        generate_button = st.button("🎨 Generate Image", type="primary", use_container_width=True, key="generate_btn")
    
    # 오른쪽 컬럼 - 생성 및 설정
    with col2:
        # 1. Generation 섹션 (맨 위)
        #st.markdown('<h2 class="section-header">🖼️ Generation</h2>', unsafe_allow_html=True)
        if "generated_image_explain" not in st.session_state: 
            st.generated_image_explain = None
        
        # 생성 버튼
        if generate_button:
            if not prompt.strip():
                st.error("❌ 프롬프트를 입력해주세요!")
            else:
                # 고급 설정에서 값들 가져오기 (session_state 사용)
                width = st.session_state.get('main_width', 1024)
                height = st.session_state.get('main_height', 1024)
                steps = st.session_state.get('main_steps', 28)
                scale = st.session_state.get('main_scale', 7.0)
                sampler = st.session_state.get('main_sampler', 'k_euler_ancestral')
                seed = st.session_state.get('main_seed', -1)
                quality_toggle = st.session_state.get('main_quality', True)
                cfg_rescale = st.session_state.get('main_cfg_rescale', 0.0)
                
                if 'nai-diffusion-4' not in model_name:
                    sm = st.session_state.get('main_sm', True)
                    sm_dyn = st.session_state.get('main_sm_dyn', False)
                else:
                    sm = True
                    sm_dyn = False
                
                # 시드 처리
                actual_seed = seed
                if seed == -1:
                    actual_seed = random.randint(0, 9999999999)
                
                # 파라미터 구성
                parameters = {
                    "width": width,
                    "height": height,
                    "n_samples": 1,
                    "seed": actual_seed,
                    "extra_noise_seed": actual_seed,
                    "sampler": sampler,
                    "steps": steps,
                    "scale": scale,
                    "negative_prompt": negative_prompt,
                    "qualityToggle": quality_toggle,
                    "cfg_rescale": cfg_rescale,
                    "noise_schedule": "native",
                    "sm": sm,
                    "sm_dyn": sm_dyn,
                    "dynamic_thresholding": False,
                    "controlnet_strength": 1.0,
                    "add_original_image": False,
                    "legacy": False,
                    "enable_hr": False,
                    "enable_AD": False
                }
                
                # 진행 표시
                with st.spinner("🎨 이미지를 생성하고 있습니다..."):
                    try:
                        start_time = time.time()
                        
                        # 이미지 생성
                        response_content = generate_nai_image(
                            access_token, prompt, model_name, parameters
                        )
                        
                        # 응답 처리
                        image = process_nai_response(response_content)
                        st.session_state.generated_image = image
                        end_time = time.time()
                        generation_time = round(end_time - start_time, 2)
                        st.session_state.generated_image_explain = {
                            "model": selected_model,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "seed": actual_seed,
                            "steps": steps,
                            "scale": scale,
                            "sampler": sampler,
                            "size": f"{width}x{height}",
                            "generation_time": f"{generation_time}s"
                        }
                    
                    except Exception as e:
                        st.error(f"❌ 생성 실패: {str(e)}")
                        st.session_state.generated_image_explain = None
                        
                        # 에러 세부 정보
                        with st.expander("🔍 Error Details"):
                            st.text(str(e))

        # 이미지 표시 및 다운로드 (생성 버튼 처리 후)
        if "generated_image" in st.session_state and st.session_state.generated_image is not None:
            generated_image = st.session_state.generated_image

            st.markdown("""
                <div style='max-height: 400px; overflow: auto; border: 1px solid #ccc; padding: 4px; border-radius: 8px'>
            """, unsafe_allow_html=True)
            st.image(generated_image, caption="Generated Image", output_format="PNG", use_container_width=True)

            # 다운로드 버튼 - 타임스탬프를 key로 사용하여 유니크하게 만들기
            img_buffer = io.BytesIO()
            generated_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            # 현재 시간을 key에 포함하여 유니크하게 만들기
            unique_key = f"download_btn_{int(time.time() * 1000)}"
            
            st.download_button(
                label="💾 Download Image",
                data=img_buffer.getvalue(),
                file_name=f"nai_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
                key=unique_key
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # 생성 정보 표시
            if "generated_image_explain" in st.session_state:
                with st.expander("📊 Generation Info"):
                    st.json(st.session_state.generated_image_explain)
                # 결과 표시
                _generation_time = st.session_state.generated_image_explain.get("generation_time", None)
                st.success(f"✅ 생성 완료! ({_generation_time}초)")

        if "generated_image" not in st.session_state:
            blank_img = Image.new("RGB", (512, 64), color=(240, 240, 240))
            st.image(blank_img, caption="No Image Generated Yet", use_container_width=True)
        # 2. Basic Settings 섹션
        st.markdown('<h2 class="section-header">⚙️ Basic Settings</h2>', unsafe_allow_html=True)
        
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            width = st.selectbox(
                "Width",
                [512, 640, 768, 832, 896, 1024, 1152, 1216],
                index=5,  # 1024 기본값
                help="이미지 너비",
                key="main_width"
            )
            
            height = st.selectbox(
                "Height", 
                [512, 640, 768, 832, 896, 1024, 1152, 1216],
                index=5,  # 1024 기본값
                help="이미지 높이",
                key="main_height"
            )
        
        with col2_2:
            steps = st.slider(
                "Steps",
                min_value=1,
                max_value=50,
                value=28,
                help="생성 스텝 수 (높을수록 품질 향상, 시간 증가)",
                key="main_steps"
            )
            
            scale = st.slider(
                "CFG Scale",
                min_value=1.0,
                max_value=30.0,
                value=7.0,
                step=0.5,
                help="프롬프트 준수도 (높을수록 프롬프트를 더 정확히 따름)",
                key="main_scale"
            )
        
        # 고급 설정
        with st.expander("🔧 Advanced Settings"):
            sampler = st.selectbox(
                "Sampler",
                SAMPLERS,
                index=1,  # k_euler_ancestral 기본값
                help="샘플링 방법",
                key="main_sampler"
            )
            
            seed = st.number_input(
                "Seed",
                min_value=-1,
                max_value=9999999999,
                value=-1,
                help="-1은 랜덤 시드",
                key="main_seed"
            )
            
            quality_toggle = st.checkbox(
                "Quality Tags",
                value=True,
                help="품질 향상 태그 자동 추가",
                key="main_quality"
            )
            
            cfg_rescale = st.slider(
                "CFG Rescale",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="CFG 재조정 (NAI V4에서 권장: 0.0)",
                key="main_cfg_rescale"
            )
            
            if 'nai-diffusion-4' not in model_name:
                sm = st.checkbox(
                    "SMEA",
                    value=True,
                    help="SMEA 활성화 (V3 모델용)",
                    key="main_sm"
                )
                
                sm_dyn = st.checkbox(
                    "SMEA DYN", 
                    value=False,
                    help="Dynamic SMEA (V3 모델용)",
                    key="main_sm_dyn"
                )
        
        # 3. Tips 섹션 (맨 아래)
        st.markdown('<h2 class="section-header">💡 Tips</h2>', unsafe_allow_html=True)
        
        with st.expander("📖 사용 가이드"):
            st.markdown("""
            **프롬프트 작성 팁:**
            - 구체적이고 상세한 설명 사용
            - 품질 관련 태그: `masterpiece, best quality, detailed`
            - 스타일 태그: `anime style, realistic, oil painting`
            - 감정 표현: `smile, happy, serious`
            
            **네거티브 프롬프트 권장:**
            - `lowres, bad quality, blurry, worst quality`
            - `bad anatomy, extra limbs, deformed`
            - `text, watermark, signature`
            
            **모델별 특징:**
            - **V3**: 범용적, 안정적
            - **V4**: 최신, 고품질, 캐릭터 특화
            - **Furry V3**: 수인 캐릭터 특화
            """)
    
    # 반응형 CSS (단순화)
    st.markdown("""
    <style>
    /* 모바일에서 컬럼을 세로로 배치 */
    @media (max-width: 768px) {
        .stHorizontalBlock > div {
            flex-direction: column !important;
        }
        .section-header {
            font-size: 1.25rem;
        }
    }
    
    /* 작은 태블릿 */
    @media (max-width: 900px) and (min-width: 769px) {
        .section-header {
            font-size: 1.3rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
