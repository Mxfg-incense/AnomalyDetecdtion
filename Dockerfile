# 使用官方Python运行时作为父镜像  
FROM python:3.10-slim
  
# 设置工作目录  
WORKDIR /server 
  
# 将当前目录内容复制到位于/app中的容器中  
COPY server /server 
  
# 安装任何所需的包  
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt 
  
# 让容器监听5000端口  
EXPOSE 5000  
  
# 运行gunicorn服务器  
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "server:app"]