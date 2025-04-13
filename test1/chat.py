import requests
import json

class DeepSeekChat:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def chat(self, message, model="deepseek-chat", temperature=0.7):
        """
        与DeepSeek聊天
        :param message: 用户输入的消息
        :param model: 使用的模型，默认为deepseek-chat
        :param temperature: 生成文本的随机性，值越大越随机
        :return: 模型返回的回复
        """
        data = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "temperature": temperature
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(data)
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    # 替换为你的DeepSeek API Key
    api_key = "sk-0f1db7510b84470ab528ac5e648be35d"
    
    chat_bot = DeepSeekChat(api_key)
    
    while True:
        user_input = input("你: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束聊天")
            break
            
        response = chat_bot.chat(user_input)
        if response:
            print(f"DeepSeek: {response}")
        else:
            print("抱歉，聊天出现错误")

