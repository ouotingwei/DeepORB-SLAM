import socket
import pickle
import cv2

def process_keypoints(keypoints):
    # 示例处理函数，假设我们只是简单地返回原始的特征点
    # 你可以在这里添加实际的处理逻辑
    return keypoints

def main():
    # 创建Socket对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定到主机和端口
    server_socket.bind(('127.0.0.1', 8888))

    # 开始监听
    server_socket.listen(5)
    print("服务器正在监听端口 8888")

    while True:
        # 接受客户端连接
        client_socket, addr = server_socket.accept()
        print(f"连接来自: {addr}")

        # 接收数据
        data = b''
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet

        # 反序列化接收到的特征点数据
        keypoints = pickle.loads(data)

        # 处理特征点数据
        processed_keypoints = process_keypoints(keypoints)

        # 序列化处理后的特征点数据
        data_to_send = pickle.dumps(processed_keypoints)

        # 发送处理后的特征点数据回客户端
        client_socket.sendall(data_to_send)

        # 关闭客户端连接
        client_socket.close()

if __name__ == "__main__":
    main()
