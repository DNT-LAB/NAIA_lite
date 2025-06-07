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
import pickle
import math
import numpy as np
from collections import defaultdict

# 페이지 설정
st.set_page_config(
    page_title="NovelAI with Smart Tags",
    page_icon="🎨",
    layout="wide"
)

# CSS 스타일링
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

.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 1rem;
    color: #4ECDC4;
}

.tag-recommendation-box {
    background-color: #f0f8f8;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #4ECDC4;
}

.recommended-tag {
    display: inline-block;
    background-color: #e8f4f8;
    padding: 0.3rem 0.6rem;
    margin: 0.2rem;
    border-radius: 0.3rem;
    border: 1px solid #4ECDC4;
    cursor: pointer;
    transition: all 0.2s;
}

.recommended-tag:hover {
    background-color: #4ECDC4;
    color: white;
}

/* 반응형 */
@media (max-width: 1200px) {
    .stHorizontalBlock > div {
        flex-direction: column !important;
    }
    .stHorizontalBlock > div > div {
        width: 100% !important;
        max-width: 100% !important;
    }
}

@media (max-width: 768px) {
    .section-header {
        font-size: 1.25rem;
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

# 태그 추천 시스템 클래스
class StreamlitTagRecommendationSystem:
    def __init__(self, model_path='recommendation_model.pkl', frequency_path='tag_frequency.json'):
        """Streamlit용 경량화된 태그 추천 시스템"""
        self.model_loaded = False
        self.error_message = None
        
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            with open(frequency_path, 'r', encoding='utf-8') as f:
                self.frequency_data = json.load(f)
            
            self.cooccurrence_matrix = self.model_data['cooccurrence_matrix']
            self.frequent_tags = set(self.model_data['frequent_tags'])
            self.tag_frequency = self.model_data['tag_frequency']
            self.total_rows = self.frequency_data.get('total_rows', 1)
            self.model_loaded = True
            
        except FileNotFoundError as e:
            self.error_message = f"모델 파일을 찾을 수 없습니다: {e}"
        except Exception as e:
            self.error_message = f"모델 로딩 오류: {e}"
    
    def get_exclude_ratio(self, iteration):
        """반복별 제외 비율"""
        if iteration == 0:
            exclude_ratio = np.random.normal(0.03, 0.02)
            exclude_ratio = np.clip(exclude_ratio, 0.01, 0.08)
        else:
            base_mean = 0.015
            base_std = 0.015
            exclude_ratio = np.random.normal(base_mean, base_std)
            exclude_ratio = np.clip(exclude_ratio, 0.005, 0.05)
        return exclude_ratio
    
    def select_tags_strategically(self, recommendations, num_select, iteration):
        """전략적 태그 선택"""
        if len(recommendations) <= num_select:
            return [tag for tag, score in recommendations]
        
        selected_tags = []
        
        if iteration == 0:  # 첫 번째: 다양성 우선
            total_recs = len(recommendations)
            top_range = min(5, total_recs // 3)
            selected_tags.extend([
                recommendations[i][0] for i in 
                np.random.choice(top_range, min(num_select//3, top_range), replace=False)
            ])
            
            mid_start = total_recs // 3
            mid_end = min(total_recs * 2 // 3, total_recs)
            if mid_end > mid_start:
                mid_range = list(range(mid_start, mid_end))
                selected_tags.extend([
                    recommendations[i][0] for i in 
                    np.random.choice(mid_range, min(num_select//3, len(mid_range)), replace=False)
                ])
            
            remaining = num_select - len(selected_tags)
            if remaining > 0:
                all_indices = list(range(total_recs))
                used_indices = [i for i, (tag, _) in enumerate(recommendations) if tag in selected_tags]
                available_indices = [i for i in all_indices if i not in used_indices]
                
                if available_indices:
                    additional = np.random.choice(
                        available_indices, 
                        min(remaining, len(available_indices)), 
                        replace=False
                    )
                    selected_tags.extend([recommendations[i][0] for i in additional])
        
        else:  # 가우시안 기반
            center = len(recommendations) * 0.3
            std = len(recommendations) * 0.4
            
            selected_indices = set()
            attempts = 0
            
            while len(selected_tags) < num_select and attempts < num_select * 10:
                attempts += 1
                idx = int(np.random.normal(center, std))
                idx = np.clip(idx, 0, len(recommendations) - 1)
                
                if idx not in selected_indices:
                    selected_indices.add(idx)
                    selected_tags.append(recommendations[idx][0])
        
        return selected_tags[:num_select]
    
    def get_recommendations_fast(self, input_tags, exclude_ratio):
        """빠른 추천 계산"""
        if not self.model_loaded:
            return []
        
        # 제외할 태그 설정
        sorted_tags = sorted(self.tag_frequency.items(), key=lambda x: x[1], reverse=True)
        top_n = int(len(sorted_tags) * exclude_ratio)
        very_common_tags = set([tag for tag, freq in sorted_tags[:top_n]])
        
        # 유효한 입력 태그만 필터링
        valid_input_tags = [tag for tag in input_tags if tag in self.frequent_tags]
        if not valid_input_tags:
            return []
        
        # 추천 점수 계산
        recommendation_scores = defaultdict(int)
        
        for input_tag in valid_input_tags:
            for (tag1, tag2), count in self.cooccurrence_matrix.items():
                if count < 3:
                    continue
                
                candidate = None
                if tag1 == input_tag and tag2 not in input_tags:
                    candidate = tag2
                elif tag2 == input_tag and tag1 not in input_tags:
                    candidate = tag1
                
                if candidate and candidate not in very_common_tags:
                    freq_weight = math.log(10000 / (self.tag_frequency.get(candidate, 1) + 1))
                    recommendation_scores[candidate] += count * freq_weight
        
        return sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    
    def fast_convergent_recommend(self, initial_tags, target_count):
        """3회 반복 경량화 추천"""
        if not self.model_loaded:
            return initial_tags
        
        current_tags = initial_tags.copy()
        initial_count = len(initial_tags)
        
        for iteration in range(3):
            needed = target_count - len(current_tags)
            if needed <= 0:
                break
            
            exclude_ratio = self.get_exclude_ratio(iteration)
            recommendations = self.get_recommendations_fast(current_tags, exclude_ratio)
            
            if not recommendations:
                break
            
            num_to_select = min(5, needed)
            selected = self.select_tags_strategically(recommendations, num_to_select, iteration)
            current_tags.extend(selected)
        
        # 최종 정리
        final_tags = sorted(list(set(current_tags)))
        if len(final_tags) > target_count:
            extra_tags = [tag for tag in final_tags if tag not in initial_tags]
            keep_extra = target_count - len(initial_tags)
            final_tags = initial_tags + extra_tags[:keep_extra]
            final_tags = sorted(final_tags)
        
        return final_tags

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

@st.cache_resource
def load_tag_recommender():
    """태그 추천 시스템 로드 (캐시됨)"""
    return StreamlitTagRecommendationSystem()

def main():
    # 태그 추천 시스템 초기화
    tag_recommender = load_tag_recommender()
    
    # 사이드바 - API 설정
    with st.sidebar:
        st.markdown('<h1 class="section-header">🎨 NAI Smart Tags</h1>', unsafe_allow_html=True)
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
            index=1,
            help="사용할 NovelAI 모델을 선택하세요"
        )
        
        model_name = MODEL_MAPPINGS[selected_model]["generate"]
        
        # 태그 추천 설정
        st.markdown('<h2 class="section-header">🏷️ Tag Recommendation</h2>', unsafe_allow_html=True)
        
        if tag_recommender.error_message:
            st.error(f"❌ {tag_recommender.error_message}")
            st.info("💡 recommendation_model.pkl과 tag_frequency.json 파일이 필요합니다.")
            use_tag_recommendation = False
        else:
            st.success("✅ 태그 추천 모델 로드됨")
            use_tag_recommendation = st.checkbox("태그 추천 사용", value=True)
            
            if use_tag_recommendation:
                target_tag_count = st.slider(
                    "목표 태그 수",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="최종 생성할 총 태그 개수"
                )
    
    # 메인 레이아웃
    col1, col2 = st.columns([1, 1])
    
    # 왼쪽 컬럼 - 프롬프트 입력 및 태그 추천
    with col1:
        st.markdown('<h2 class="section-header">📝 Prompt & Tags</h2>', unsafe_allow_html=True)
        
        # 태그 추천 기능
        if use_tag_recommendation and tag_recommender.model_loaded:
            st.markdown('<div class="tag-recommendation-box">', unsafe_allow_html=True)
            st.markdown("**🎯 스마트 태그 추천**")
            
            # 초기 태그 입력
            initial_tags_input = st.text_input(
                "기본 태그들 (쉼표로 구분)",
                placeholder="1girl, long hair, anime style",
                help="이 태그들을 기반으로 연관 태그를 추천합니다"
            )
            
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                recommend_button = st.button("🎲 태그 추천", type="secondary")
            with col_rec2:
                apply_button = st.button("✅ 적용", type="primary")
            
            # 태그 추천 실행
            if recommend_button and initial_tags_input.strip():
                initial_tags = [tag.strip() for tag in initial_tags_input.split(',') if tag.strip()]
                
                with st.spinner("태그 추천 중..."):
                    try:
                        recommended_tags = tag_recommender.fast_convergent_recommend(
                            initial_tags, target_tag_count
                        )
                        st.session_state.recommended_tags = recommended_tags
                        st.session_state.initial_tags = initial_tags
                    except Exception as e:
                        st.error(f"추천 실패: {e}")
            
            # 추천 결과 표시
            if "recommended_tags" in st.session_state:
                recommended_tags = st.session_state.recommended_tags
                initial_tags = st.session_state.get.initial_tags, [])
                
                st.markdown("**📋 추천 결과:**")
                
                # 초기 태그와 추천 태그 구분 표시
                initial_display = [f"**{tag}**" for tag in initial_tags]
                new_tags = [tag for tag in recommended_tags if tag not in initial_tags]
                
                all_display_tags = initial_display + new_tags
                tags_text = ", ".join(all_display_tags)
                st.markdown(f"*({len(recommended_tags)}개 태그)*")
                st.text_area("", value=", ".join(recommended_tags), height=100, key="recommended_display")
                
                # 프롬프트에 적용
                if apply_button:
                    st.session_state.apply_recommended_prompt = ", ".join(recommended_tags)
                    st.success("✅ 추천 태그가 프롬프트에 적용되었습니다!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 프롬프트 입력
        default_prompt = st.session_state.get('apply_recommended_prompt', '')
        prompt = st.text_area(
            "Prompt",
            value=default_prompt,
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
        if "generated_image_explain" not in st.session_state: 
            st.session_state.generated_image_explain = None
        
        # 생성 버튼 처리
        if generate_button:
            if not prompt.strip():
                st.error("❌ 프롬프트를 입력해주세요!")
            else:
                # 파라미터 수집
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
                actual_seed = seed if seed != -1 else random.randint(0, 9999999999)
                
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
                
                # 생성 실행
                with st.spinner("🎨 이미지를 생성하고 있습니다..."):
                    try:
                        start_time = time.time()
                        
                        response_content = generate_nai_image(
                            access_token, prompt, model_name, parameters
                        )
                        
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

        # 이미지 표시
        if "generated_image" in st.session_state and st.session_state.generated_image is not None:
            generated_image = st.session_state.generated_image

            st.markdown("""
                <div style='max-height: 400px; overflow: auto; border: 1px solid #ccc; padding: 4px; border-radius: 8px'>
            """, unsafe_allow_html=True)
            st.image(generated_image, caption="Generated Image", output_format="PNG", use_container_width=True)

            # 다운로드 버튼
            img_buffer = io.BytesIO()
            generated_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)

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

            # 생성 정보
            if st.session_state.generated_image_explain:
                with st.expander("📊 Generation Info"):
                    st.json(st.session_state.generated_image_explain)
                
                _generation_time = st.session_state.generated_image_explain.get("generation_time", None)
                st.success(f"✅ 생성 완료! ({_generation_time})")

        else:
            blank_img = Image.new("RGB", (512, 64), color=(240, 240, 240))
            st.image(blank_img, caption="No Image Generated Yet", use_container_width=True)
        
        # 기본 설정
        st.markdown('<h2 class="section-header">⚙️ Basic Settings</h2>', unsafe_allow_html=True)
        
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            width = st.selectbox(
                "Width",
                [512, 640, 768, 832, 896, 1024, 1152, 1216],
                index=5,
                key="main_width"
            )
            
            height = st.selectbox(
                "Height", 
                [512, 640, 768, 832, 896, 1024, 1152, 1216],
                index=5,
                key="main_height"
            )
        
        with col2_2:
            steps = st.slider(
                "Steps",
                min_value=1,
                max_value=50,
                value=28,
                key="main_steps"
            )
            
            scale = st.slider(
                "CFG Scale",
                min_value=1.0,
                max_value=30.0,
                value=7.0,
                step=0.5,
                key="main_scale"
            )
        
        # 고급 설정
        with st.expander("🔧 Advanced Settings"):
            sampler = st.selectbox(
                "Sampler",
                SAMPLERS,
                index=1,
                key="main_sampler"
            )
            
            seed = st.number_input(
                "Seed",
                min_value=-1,
                max_value=9999999999,
                value=-1,
                key="main_seed"
            )
            
            quality_toggle = st.checkbox(
                "Quality Tags",
                value=True,
                key="main_quality"
            )
            
            cfg_rescale = st.slider(
                "CFG Rescale",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="main_cfg_rescale"
            )
            
            if 'nai-diffusion-4' not in model_name:
                sm = st.checkbox(
                    "SMEA",
                    value=True,
                    key="main_sm"
                )
                
                sm_dyn = st.checkbox(
                    "SMEA DYN", 
                    value=False,
                    key="main_sm_dyn"
                )
        
        # Tips
        st.markdown('<h2 class="section-header">💡 Tips</h2>', unsafe_allow_html=True)
        
        with st.expander("📖 사용 가이드"):
            st.markdown("""
            **🎯 스마트 태그 추천 사용법:**
            1. 기본 태그 입력 (예: "1girl, long hair")
            2. "태그 추천" 버튼 클릭
            3. 추천된 태그 확인 후 "적용" 클릭
            4. 프롬프트가 자동으로 완성됨
            
            **📝 프롬프트 작성 팁:**
            - 구체적이고 상세한 설명 사용
            - 품질 관련 태그: `masterpiece, best quality, detailed`
            - 스타일 태그: `anime style, realistic, oil painting`
            
            **🚫 네거티브 프롬프트 권장:**
            - `lowres, bad quality, blurry, worst quality`
            - `bad anatomy, extra limbs, deformed`
            - `text, watermark, signature`
            
            **🤖 모델별 특징:**
            - **V3**: 범용적, 안정적
            - **V4**: 최신, 고품질, 캐릭터 특화
            - **Furry V3**: 수인 캐릭터 특화
            """)

if __name__ == "__main__":
    main()
