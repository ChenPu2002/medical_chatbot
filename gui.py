"""GUI module for ChatDoctor application."""
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QLineEdit, QHBoxLayout, QComboBox, QLabel
)
from PyQt5.QtGui import QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, pyqtSlot, QTimer

print("Loading GUI...")
# pylint: disable=wildcard-import,unused-wildcard-import,wrong-import-position
from middleware import TreePredictor, APIPredictor
print("Loading Model...")


class ChatWindow(QMainWindow):
    """Main window for the ChatDoctor application."""
    def __init__(self):
        super().__init__()
        self.roundcount = 0
        self.state = "Tree"
        self.tree_state = None
        self.user_message = ""  # Initialize user_message attribute

        self.setWindowTitle('ChatDoctor - Your Personal Health Assistant')
        # self.setFixedSize(800, 600)

        # Initialize QWebEngineView
        self.view = QWebEngineView()

        # Set the initial HTML content
        self.init_ui()
        self.view.loadFinished.connect(self.welcome_message)

        # Create a vertical layout for the main window
        main_layout = QVBoxLayout()

        # Create a title
        self.label = QLabel()
        self.label.setText("ChatDoctor")
        self.label.setFont(QFont("Verdana", 40))
        self.label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label)

        # Create a horizontal layout for the top controls
        top_layout = QHBoxLayout()

        # Model selection dropdown
        self.model_selection = QComboBox()
        self.model_selection.addItems(["Tree", "API"])
        self.model_selection.currentTextChanged.connect(self.on_combobox_changed)

        # Reset button to clear chat content
        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet("""
                background-color: green; 
                color: white;                
                border-radius: 10px; /* Rounded corners */
                padding: 5px;
                font-size: 16px;
                min-width: 100px; /* Minimum width of 100 pixels */
                """)
        self.reset_button.clicked.connect(self.reset_chat)

        # Add model selection and reset button to the top layout
        top_layout.addWidget(QLabel("Model: "))
        top_layout.addWidget(self.model_selection)
        top_layout.addStretch()  # This pushes the model selection and reset button apart
        top_layout.addWidget(self.reset_button)

        # Input box for sending messages
        self.input = QLineEdit()
        self.input.setStyleSheet("""
            QLineEdit {
                font-size: 16px; /* Larger font size */
                padding: 5px; /* Padding inside the input box */
                min-height: 30px; /* Minimum height */
            }
        """)
        self.input.setPlaceholderText("Type your message here")
        self.input.returnPressed.connect(self.tree_send_message)

        # Send button to submit messages
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: green;
                color: white;
                border-radius: 10px;
                padding: 5px;
                font-size: 16px;
                min-height: 30px;
            }
        """)
        self.send_button.clicked.connect(self.tree_send_message)

        # Add the chat view and controls to the main layout
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.view)
        main_layout.addWidget(self.input)
        main_layout.addWidget(self.send_button)

        # Set the central widget of the window to our layout
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        self.disease_predictor = TreePredictor()
    @pyqtSlot(str)
    def on_combobox_changed(self, state):
        """Handle model selection change."""
        self.state = state
        # model type choose and change the object to connect
        if self.state == "Tree":
            self.disease_predictor = TreePredictor()
            # Safely disconnect API signals if they exist
            try:
                self.input.returnPressed.disconnect(self.api_send_message)
                self.send_button.clicked.disconnect(self.api_send_message)
            except TypeError:
                # Signals weren't connected, which is fine
                pass
            self.input.returnPressed.connect(self.tree_send_message)
            self.send_button.clicked.connect(self.tree_send_message)
        elif self.state == "API":
            self.disease_predictor = APIPredictor()
            # Safely disconnect tree signals if they exist
            try:
                self.input.returnPressed.disconnect(self.tree_send_message)
                self.send_button.clicked.disconnect(self.tree_send_message)
            except TypeError:
                # Signals weren't connected, which is fine
                pass
            self.input.returnPressed.connect(self.api_send_message)
            self.send_button.clicked.connect(self.api_send_message)

        self.reset_chat()

    def init_ui(self):
        """Initialize the chat UI with HTML."""
        # Load initial HTML for the chat interface
        initial_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    overflow: hidden;
                }
                .chat-container {
                    display: flex;
                    flex-direction: column;
                    padding: 10px;
                    min-height: 300px;
                    max-height: 100vh;
                    overflow-y: auto;
                    scroll-behavior: smooth;
                    background-color: white;
                    box-sizing: border-box;
                }
                .chat-message {
                    max-width: 60%;
                    margin: 5px;
                    padding: 10px;
                    border-radius: 15px;
                }
                .user-message {
                    align-self: flex-end;
                    background-color: #AED6F1;
                }
                .bot-message {
                    align-self: flex-start;
                    background-color: #ABEBC6;
                }
            </style>
        </head>
        <body>
            <div id="chat" class="chat-container">
                <!-- Chat messages will be added here -->
            </div>
            <script>
                function addMessage(sender, message) {
                    var chatContainer = document.getElementById('chat');
                    var messageDiv = document.createElement('div');
                    messageDiv.classList.add('chat-message');
                    messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
                    messageDiv.innerText = message;
                    chatContainer.appendChild(messageDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            </script>
        </body>
        </html>
        """
        self.view.setHtml(initial_html)

    def tree_send_message(self):
        """Handle sending messages in Tree model mode."""
        # Get the user's message from the input box
        user_message = self.input.text().strip()
        # for empty input, pass
        if user_message == "":
            return

        # for the first round, check the state
        if self.roundcount == 0 and user_message != "exit":
            self.input.clear()
            self.view.page().runJavaScript(f"addMessage('user', `{user_message}`);")
            if user_message == "1":
                self.tree_state = 1
                msg = "Please briefly describe your illness (in one word)."
                self.view.page().runJavaScript(f"addMessage('bot', `{msg}`);")
            elif user_message == "2":
                self.tree_state = 2
                msg = "Please input the name of medicine."
                self.view.page().runJavaScript(f"addMessage('bot', `{msg}`);")
            elif user_message == "exit":
                self.view.page().runJavaScript("addMessage('bot', `Goodbye!`);")
                self.view.page().runJavaScript("addMessage('bot', `Exiting...`);")
                # time.sleep(1)
                QTimer.singleShot(1000, self.reset_chat)
            else:
                # invalid input, ask again
                msg = "Invalid input, please input number 1 or 2."
                self.view.page().runJavaScript(f"addMessage('bot', `{msg}`);")
                self.roundcount -= 1
        # elif self.roundcount == 0 and user_message == "exit":
        #     self.input.clear()
        #     self.view.page().runJavaScript(f"addMessage('user', `exit`);")
        #     self.view.page().runJavaScript(f"addMessage('bot', `Goodbye!`);")

        # when roundcount > 0, the control will be passed to the model
        elif self.roundcount > 0 and user_message:
            # Clear the input box after sending the message
            self.input.clear()
            # Add the user's message to the chat interface
            restart_default = "RESTART_DEFAULT"
            if (self.tree_state == 1 and self.roundcount == 2
                    and user_message == restart_default):
                pass
            else:
                script = f"addMessage('user', `{user_message}`);"
                self.view.page().runJavaScript(script)

            if self.tree_state == 1 and self.roundcount == 2 and user_message == "0":
                self.roundcount -= 1
                msg = "Please input the name of symptom again."
                self.view.page().runJavaScript(f"addMessage('bot', `{msg}`);")
                self.user_message = restart_default
                self.disease_predictor.count = 0
                self.tree_send_message()
                return

            if self.tree_state == 1:
                self.disease_predictor.disease_or_meds = 1
                # get user_message here and call the model to get the bot_message
                bot_message = self.disease_predictor.get_response(user_message)
                script = f"addMessage('bot', `{bot_message}`);"
                self.view.page().runJavaScript(script)
                if user_message == "exit":
                    self.view.page().runJavaScript("addMessage('bot', `Exiting...`);")
                    # time.sleep(1)
                    QTimer.singleShot(1000, self.reset_chat)

            elif self.tree_state == 2:
                self.disease_predictor.disease_or_meds = 2

                bot_message = self.disease_predictor.get_response(user_message)
                script = f"addMessage('bot', `{bot_message}`);"
                self.view.page().runJavaScript(script)
                if user_message == "exit":
                    self.view.page().runJavaScript("addMessage('bot', `Exiting...`);")
                    # time.sleep(1)
                    QTimer.singleShot(1000, self.reset_chat)
        self.roundcount += 1

    def api_send_message(self):
        """Handle sending messages in API model mode."""
        # Get the user's message from the input box
        user_message = self.input.text().strip()

        if user_message:
            # Clear the input box after sending the message
            self.input.clear()
            # Add the user's message to the chat interface
            script = f"addMessage('user', `{user_message}`);"
            self.view.page().runJavaScript(script)
            # get user_message here and call the model to get the bot_message
            bot_message = self.disease_predictor.get_response(user_message)
            script = f"addMessage('bot', `{bot_message}`);"
            self.view.page().runJavaScript(script)
            if user_message == "exit":
                self.view.page().runJavaScript("addMessage('bot', `Exiting...`);")
                # time.sleep(1)
                QTimer.singleShot(1000, self.reset_chat)

    def welcome_message(self):
        """Display the welcome message."""
        if self.state == "Tree":
            welcome = ("Hello! I'm ChatDoctor, your personal health assistant. "
                       "Please choose what kind of question you want to ask.\n\n"
                       " 1) Describe your illness\n"
                       " 2) Check the uses and side effects of medicines\n\n"
                       " Type 1 or 2 to continue. \n\n"
                       " Type 'exit' or click Reset button to continue at any time.")
        elif self.state == "API":
            welcome = ("Hello! I'm ChatDoctor, your personal health assistant. "
                       "Please input your question to get a response.")
        script = f"addMessage('bot', `{welcome}`);"
        self.view.page().runJavaScript(script)

    def reset_chat(self):
        """Reset the chat state."""
        # Clear all messages from the chat interface
        # constant used for GUI to check the current state of the chat
        # after roundcount > 0, the control of thread will be passed to the model
        self.roundcount = 0
        self.disease_predictor.count = 0
        # self.disease_predictor = TreePredictor()
        self.init_ui()

if __name__ == '__main__':
    raise ValueError("This file is not meant to be run directly. Run main.py instead.")
