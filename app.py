import os
from typing import List, Tuple
import cv2
import numpy as np
import streamlit as st
from tqdm import tqdm
from PIL import Image, ImageOps

from core.cropping import (
    ensure_dir,
    grid_tiles,
    grid_tiles_strict,
    draw_rects,
    save_crop,
    maybe_save_full,
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def list_images(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    files = []
    for fn in os.listdir(input_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext in SUPPORTED_EXTS:
            files.append(os.path.join(input_dir, fn))
    files.sort()
    return files

def load_bgr(path: str) -> np.ndarray:
    # 优先 Pillow 读取并处理 EXIF 方向
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            rgb = np.array(im)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img

def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def read_uploaded_to_bgr(uploaded_file):
    """
    将 Streamlit 上传的文件读取为 BGR 图像（OpenCV 格式）
    """
    data = uploaded_file.read()
    uploaded_file.seek(0)  # 重置指针，避免后续再次读取失败
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法解码上传的图片：{uploaded_file.name}")
    return bgr

def main():
    st.set_page_config(page_title="图片裁切工具", layout="wide")
    st.title("图片裁切工具")

    # 侧边栏设置
    st.sidebar.header("参数设置")

    default_in = os.path.join(os.getcwd(), "input_images")
    default_out = os.path.join(os.getcwd(), "output_crops")

    # 新增：输入来源（文件夹 / 上传文件）
    input_source = st.sidebar.radio("输入来源", options=["文件夹", "上传文件"], index=0)

    if input_source == "文件夹":
        input_dir = st.sidebar.text_input("输入目录", value=default_in)
        uploaded_files = []
    else:
        input_dir = None
        uploaded_files = st.sidebar.file_uploader(
            "上传图片（可多选）", type=["png", "jpg", "jpeg", "bmp", "webp"], accept_multiple_files=True
        )

    output_dir = st.sidebar.text_input("输出目录", value=default_out)
    output_dir = ensure_dir(output_dir) or output_dir

    # 裁切尺寸预设
    preset_mode = st.sidebar.selectbox("裁切尺寸预设", options=["自定义尺寸", "按边长比例", "按数量估算"], index=0)
    size_ratio = None
    desired_k = None
    
    if preset_mode == "按边长比例":
        size_ratio = st.sidebar.slider("边长比例（相对原图宽高）", 0.05, 0.8, 0.10, 0.01)
    elif preset_mode == "按数量估算":
        desired_k = st.sidebar.number_input("期望每图裁切数量（用于估算边长）", min_value=1, max_value=1000, value=10, step=1)
    
    # 新增：窗口尺寸是否跟随原图比例（主要影响“自定义尺寸”模式）
    keep_aspect = st.sidebar.checkbox("窗口尺寸跟随原图比例", value=True, help="保持裁切窗口与原图相同的宽高比例，避免变形")
    
    # 自定义尺寸参数（仅在自定义模式下显示）
    if preset_mode == "自定义尺寸":
        col_size = st.sidebar.columns(2)
        with col_size[0]:
            target_w = st.number_input("裁切宽度（px）", min_value=64, max_value=8192, value=768, step=32)
        with col_size[1]:
            target_h = st.number_input("裁切高度（px）", min_value=64, max_value=8192, value=1024, step=32)
    else:
        target_w, target_h = 768, 1024  # 预设模式下的默认值（实际不会使用）

    # 网格参数
    grid_overlap = st.sidebar.slider("重叠比例", 0.0, 0.9, 0.2, 0.05)
    
    # 新增：严格模式开关（默认关闭以优先覆盖全图）
    strict_mode = st.sidebar.checkbox(
        "严格模式（只导出完整尺寸的裁片）",
        value=False,
        help="开启后按规则网格生成，可能因步长与图像尺寸不整除导致未覆盖。若出现底部/右侧未覆盖，请关闭严格模式或增大重叠比例。"
    )

    # 导出设置
    st.sidebar.markdown("---")
    resize_crop = st.sidebar.selectbox(
        "裁片导出尺寸", 
        options=["保持裁片尺寸", "缩放到目标尺寸"], 
        index=0
    )
    no_resize = st.sidebar.checkbox("严格不缩放（保留原始像素）", value=True)

    output_dir = ensure_dir(output_dir) or output_dir

    # 主面板
    # 主面板数据源
    if input_source == "文件夹":
        imgs = list_images(input_dir)
    else:
        imgs = list(uploaded_files or [])

    left, right = st.columns([1, 2])

    with left:
        st.subheader("图片列表")
        if not imgs:
            st.info("请在左侧选择输入来源并提供图片（选择目录或上传文件）")
        else:
            if input_source == "文件夹":
                img_names = [os.path.basename(p) for p in imgs]
            else:
                img_names = [f.name for f in imgs]
            sel_name = st.selectbox("选择图片", options=img_names, index=0)
            sel_idx = img_names.index(sel_name)

            if input_source == "文件夹":
                sel_path = imgs[sel_idx]
            else:
                sel_file = imgs[sel_idx]

    if imgs:
        # 读取选中图片
        if input_source == "文件夹":
            bgr = load_bgr(sel_path)
        else:
            bgr = read_uploaded_to_bgr(sel_file)

        H, W = bgr.shape[:2]
        st.write(f"分辨率: {W}x{H}")

        # 根据预设计算本图的窗口大小
        def compute_window_size(W: int, H: int) -> Tuple[int, int]:
            if preset_mode == "按边长比例" and size_ratio is not None:
                # 本模式天然保持原图比例
                ww = max(1, int(W * float(size_ratio)))
                hh = max(1, int(H * float(size_ratio)))
                return ww, hh
            elif preset_mode == "按数量估算" and desired_k is not None and desired_k > 0:
                # 本模式也天然保持原图比例
                side_ratio = max(0.01, min(0.95, (1.0 / float(desired_k)) ** 0.5))
                ww = max(1, int(W * side_ratio))
                hh = max(1, int(H * side_ratio))
                return ww, hh
            else:
                # 自定义尺寸
                if keep_aspect:
                    # 按原图比例用宽度推算高度
                    ww = int(target_w)
                    hh = max(1, int(round(ww * H / float(W))))
                    return ww, hh
                else:
                    return int(target_w), int(target_h)

        # 生成裁切建议
        if st.button("生成裁切建议"):
            win_w, win_h = compute_window_size(W, H)
            
            # 根据严格模式选择函数
            if strict_mode:
                rects = grid_tiles_strict((H, W), win_w, win_h, overlap=float(grid_overlap))
            else:
                rects = grid_tiles((H, W), win_w, win_h, overlap=float(grid_overlap))
            
            st.session_state["rects"] = rects
            st.session_state["win_size"] = (win_w, win_h)
            st.session_state["strict_mode"] = strict_mode

        with right:
            st.subheader("预览")
            st.image(to_rgb(bgr), caption="原图", use_column_width=True)

            if "rects" in st.session_state and len(st.session_state["rects"]) > 0:
                rects = st.session_state["rects"]
                overlay = draw_rects(bgr, rects)
                
                if "win_size" in st.session_state:
                    win_w, win_h = st.session_state["win_size"]
                    mode_text = "严格模式" if st.session_state.get("strict_mode", False) else "普通模式"
                    st.caption(f"窗口尺寸: {win_w}x{win_h} | 重叠比例: {grid_overlap:.2f} | {mode_text} | 共 {len(rects)} 个裁片")
                
                st.image(to_rgb(overlay), caption="裁切建议叠加", use_column_width=True)

                # 缩略图网格
                st.subheader("裁切缩略图")
                if strict_mode:
                    st.write("🔒 严格模式：所有裁片均为完整尺寸")
                else:
                    st.write("⚠️ 普通模式：边缘裁片可能尺寸不足")
                
                cols = st.columns(4)
                for i, (x, y, w, h) in enumerate(rects):
                    crop = bgr[y:y+h, x:x+w]
                    cols[i % 4].image(to_rgb(crop), caption=f"片段-{i} ({w}x{h})", use_column_width=True)

        # 批量导出
        st.markdown("---")
        if st.button("批量导出所有图片的裁切结果"):
            # 确定导出设置
            if no_resize:
                resize_crop_to = None
            else:
                resize_crop_to = (int(target_w), int(target_h)) if resize_crop == "缩放到目标尺寸" else None

            pbar = st.progress(0)
            log = st.empty()
            
            total_crops = 0
            failed_images = []

            for idx, item in enumerate(tqdm(imgs)):
                # 读取每张图片与命名
                if input_source == "文件夹":
                    path = item
                    base = os.path.splitext(os.path.basename(path))[0]
                    bgr_i = load_bgr(path)
                    H_i, W_i = bgr_i.shape[:2]
                    # 原图使用原文件名保存
                    full_out = os.path.join(output_dir, os.path.basename(path))
                else:
                    uf = item  # UploadedFile
                    base = os.path.splitext(os.path.basename(uf.name))[0]
                    bgr_i = read_uploaded_to_bgr(uf)
                    H_i, W_i = bgr_i.shape[:2]
                    # 原图使用上传的原文件名保存
                    full_out = os.path.join(output_dir, uf.name)

                maybe_save_full(bgr_i, full_out, resize_to=None)

                # 按预设为该图片计算窗口
                def compute_window_size_i(Wi: int, Hi: int) -> Tuple[int, int]:
                    if preset_mode == "按边长比例" and size_ratio is not None:
                        ww = max(1, int(Wi * float(size_ratio)))
                        hh = max(1, int(Hi * float(size_ratio)))
                        return ww, hh
                    elif preset_mode == "按数量估算" and desired_k is not None and desired_k > 0:
                        side_ratio = max(0.01, min(0.95, (1.0 / float(desired_k)) ** 0.5))
                        ww = max(1, int(Wi * side_ratio))
                        hh = max(1, int(Hi * side_ratio))
                        return ww, hh
                    else:
                        if keep_aspect:
                            ww = int(target_w)
                            hh = max(1, int(round(ww * Hi / float(Wi))))
                            return ww, hh
                        else:
                            return int(target_w), int(target_h)

                win_w_i, win_h_i = compute_window_size_i(W_i, H_i)

                # 生成裁切（根据严格模式）
                if strict_mode:
                    rects_export = grid_tiles_strict((H_i, W_i), win_w_i, win_h_i, float(grid_overlap))
                else:
                    rects_export = grid_tiles((H_i, W_i), win_w_i, win_h_i, float(grid_overlap))
                
                # 保存裁片
                for i, r in enumerate(rects_export):
                    out_path = os.path.join(output_dir, f"{base}_crop_{i}.jpg")
                    save_crop(bgr_i, r, out_path, resize_to=resize_crop_to)

                total_crops += len(rects_export)
                pbar.progress((idx + 1) / len(imgs))
                log.text(f"导出 {idx + 1}/{len(imgs)}: {base} ({len(rects_export)} 个裁片)")

            mode_text = "严格模式" if strict_mode else "普通模式"
            st.success(f"导出完成！{mode_text}下共处理 {len(imgs)} 张图片，生成 {total_crops} 个裁片，结果保存在：{output_dir}")

if __name__ == "__main__":
    main()