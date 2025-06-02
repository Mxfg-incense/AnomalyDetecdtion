# 使用官方Python运行时作为父镜像  
FROM python:3.10-slim
  
WORKDIR /server
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY server/requirements.txt  /server/
RUN pip install --no-cache-dir -r requirements.txt 
# 设置工作目录  

# 将当前目录内容复制到位于/app中的容器中  
COPY server /server 

# 安装任何所需的包  
  
# 让容器监听5000端口  
EXPOSE 5000  
  
# 运行gunicorn服务器  
CMD ["gunicorn", "--timeout", "300", "-w", "1", "-b", "0.0.0.0:5000", "server:app"]