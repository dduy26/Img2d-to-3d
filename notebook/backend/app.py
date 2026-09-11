from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import shutil
import os
import numpy as np

# Import ĐÚNG tên Class từ các file của em
from quality_gate import QualityGate
from engine_triposr import TripoSREngine

app = FastAPI(title="2D to 3D Generation API")

os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# KHỞI TẠO CLASS (Chỉ chạy 1 lần lúc bật server để tối ưu tốc độ)
q_gate = QualityGate()
triposr_engine = TripoSREngine()

@app.post("/generate-3d/")
async def generate_3d(file: UploadFile = File(...)):
    try:
        # 1. Lưu ảnh gốc
        input_path = f"temp_uploads/{file.filename}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # --- 2. TẠO DỮ LIỆU GIẢ LẬP (MOCK DATA) TỪ DUST3R ---
        # Vì chúng ta chưa ghép DUST3R, thầy tạo dữ liệu giả để test Quality Gate
        mock_poses = [np.eye(4), np.eye(4)] # Giả lập 2 camera có góc trùng nhau (Góc = 0 độ)
        mock_confidence = np.array([0.9, 0.8])
        mock_loss = 1.5
        
        # 3. Qua cổng kiểm định chất lượng (Truyền đúng 3 tham số)
        is_high_quality, reason = q_gate.evaluate(mock_poses, mock_confidence, mock_loss)
        
        # Đổi đuôi file sang .glb theo đúng chuẩn đồ án
        output_model_path = f"outputs/result_{file.filename.split('.')[0]}.glb"
        
        # 4. Phân luồng AI
        if is_high_quality:
            print(f"Ảnh nét (Vượt qua Q-Gate) -> Gọi DUST3R")
            # Tạm dùng TripoSR để nghiệm thu
            success, model_path, exec_time = triposr_engine.run_fallback(input_path, output_model_path)
        else:
            print(f"Kích hoạt Cứu hộ (Lý do: {reason}) -> Gọi TripoSR")
            success, model_path, exec_time = triposr_engine.run_fallback(input_path, output_model_path)
            
        return {
            "status": "success",
            "quality_passed": is_high_quality,
            "gate_reason": reason,
            "execution_time_seconds": round(exec_time, 2), # Ghi nhận thời gian
            "output_file": model_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)