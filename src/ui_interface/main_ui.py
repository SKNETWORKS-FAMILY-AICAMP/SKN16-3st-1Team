import gradio as gr
import torch
import torch.nn as nn
import random


def main_ui():
    print("여기다가 코드를 넣어주세요")

    # 코드와 설명 (이전과 동일)
    code_lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "class SimpleCNN(nn.Module):",
        "    def __init__(self):",
        "        super(SimpleCNN, self).__init__()",
        "        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)",
        "        self.relu = nn.ReLU()",
        "        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)",
        "        self.fc = nn.Linear(32*14*14, 10)",
        "",
        "    def forward(self, x):",
        "        x = self.pool(self.relu(self.conv1(x)))",
        "        x = x.view(x.size(0), -1)",
        "        x = self.fc(x)",
        "        return x"
    ]

    explanations = [
        "PyTorch 라이브러리는 딥러닝을 위한 텐서 연산 및 신경망 컴포넌트를 제공합니다. 이 줄은 PyTorch 기본 모듈을 임포트합니다.",
        "torch.nn은 신경망 층(레이어)과 손실함수 등 모델 구성요소가 정의된 모듈입니다. CNN 구현에 필수입니다.",
        "빈 줄로 코드 가독성 및 구조 구분에 도움을 줍니다.",
        "SimpleCNN 클래스는 nn.Module을 상속해 사용자 정의 신경망 모델을 만듭니다. CNN 구조와 학습 함수를 포함할 예정입니다.",
        "__init__ 메서드는 클래스 생성자입니다. 여기서 레이어들을 초기화하고 각 계층의 파라미터를 정의합니다.",
        "부모 클래스(nn.Module)의 초기화 함수를 호출하여 PyTorch 내부 초기화 메커니즘을 수행합니다.",
        "첫 번째 합성곱 층으로 입력 채널 수 1(grayscale 이미지), 출력 채널 수 32, 3x3 필터, 스트라이드 1, 패딩 1을 설정합니다. 이는 입력과 출력 크기를 동일하게 유지합니다.",
        "ReLU 활성화 함수는 비선형성을 추가하여 신경망이 복잡한 함수 근사를 가능하게 합니다. 주로 음수 값을 0으로 처리합니다.",
        "MaxPooling은 공간 정보를 줄이고 계산량을 감소시키는 역할을 합니다. 2x2 필터, 스트라이드 2로 입력 크기를 절반으로 줄입니다.",
        "완전 연결층(Fully Connected Layer)은 CNN 후반부에서 고차원 특징을 기반으로 최종 클래스 10개를 분류합니다.",
        "빈 줄로 이해 및 시각적 분리 역할을 합니다.",
        "forward 메서드는 순전파 함수입니다. 입력 텐서 x가 모델을 통과할 때 연산 순서를 정의합니다.",
        "첫 번째 합성곱 층 → ReLU → 풀링 연산 순으로 특징을 추출합니다.",
        "특징 맵 형태의 텐서를 (배치 크기, 특징 수)로 평탄화하여 FC 층에 전달할 준비를 합니다.",
        "평탄화된 벡터를 통해 FC 층으로부터 클래스별 점수를 산출합니다.",
        "출력된 벡터(클래스 점수)를 반환하여 예측 결과를 만듭니다."
    ]

    full_code = "\n".join(code_lines)

    # CNN 관련 답변 후보 사전
    cnn_answers = [
        "CNN은 Convolutional Neural Network의 약자로, 이미지 처리에 강력한 딥러닝 모델입니다.",
        "합성곱 층은 필터를 이용해 입력 이미지의 특징을 추출합니다.",
        "ReLU는 음수 부분을 0으로 만들어 비선형성을 추가하는 활성화 함수입니다.",
        "MaxPooling은 특징 맵 크기를 줄여 계산량을 감소시키고 과적합을 방지합니다.",
        "완전 연결층은 최종 클래스 분류를 위해 특징 벡터를 사용합니다."
    ]

    # 질문 키워드별 매핑
    keyword_answer_map = {
        "CNN": cnn_answers,
        "합성곱": [cnn_answers[1]],
        "ReLU": [cnn_answers[2]],
        "풀링": [cnn_answers[3]],
        "완전연결": [cnn_answers[4]],
    }

    def chatbot_fn(question, history):
        history = history or []
        matched_answers = []
        for kw, answers in keyword_answer_map.items():
            if kw in question:
                matched_answers.extend(answers)
        if matched_answers:
            # 여러 개일 경우 랜덤 한 개 또는 전체 중 첫 개 선택
            answer = random.choice(matched_answers)
        else:
            answer = "죄송해요, 아직 그 질문에 대한 답변은 준비되지 않았어요."
        history.append((question, answer))
        return history, history

    def line_explain(idx):
        return explanations[int(idx)]

    with gr.Blocks() as demo:
        gr.Markdown("## 🧠 CNN 설명 챗봇 + 줄 단위 설명 (코드 한 줄 클릭!)")
        with gr.Column():
            msg = gr.Textbox(label="질문 입력", placeholder="질문을 입력하세요.")
            chatbot = gr.Chatbot(label="QnA 챗봇")
            clear = gr.Button("대화 초기화")
            msg.submit(chatbot_fn, [msg, chatbot], [chatbot, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        with gr.Row():
            with gr.Column():
                with gr.Accordion("📂 코드 보기 (열고 닫기 가능)", open=False):
                    code_out = gr.Code(value=full_code, label="예제 코드", language="python", interactive=False)
                    line_selector = gr.Dropdown(
                        choices=[f"{i+1}: {line}" for i, line in enumerate(code_lines)],
                        label="코드 한 줄 선택(설명 보기)",
                        type="index"
                    )
                    line_exp = gr.Textbox(label="선택된 줄 설명", interactive=False)
                    line_selector.change(fn=line_explain, inputs=line_selector, outputs=line_exp)

    demo.launch(share=True, debug=True)    


if __name__ == "__main__":
    main_ui()    