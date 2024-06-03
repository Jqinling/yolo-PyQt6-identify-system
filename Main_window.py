from PyQt6 import QtWidgets,QtCore,QtGui
from threading import Thread
import cv2,os,time
import sys

#关闭yolo详细日志
os.environ['YOLO_VERBOSE'] = 'False' #这段代码得写导入ultralytics之前,不然不生效
from ultralytics import YOLO


class Main_Window(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()

        self.setupUI()

        #设置全局变量，改BUG！
        self.flag_camera = False
        self.flag_video = False
        self.flag_image = False
        #默认模式为目标检测
        self.flag_mode_det = True

        #默认模型大小为n（纳米）
        self.model_scale = 0

        self.bottomLayout.addLayout(self.btnLayout)
        #设置定时器
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        self.timer_video = QtCore.QTimer()
        self.timer_video.timeout.connect(self.show_video)

        # 加载模型
        self.model = YOLO("model\yolov8n.pt")

        #放置待处理的图片
        self.frameToanalyze_camera = []
        self.frameToanalyze_video = []

        #启动处理帧独立线程
        Thread(target=self.frameAnalyzeThreadFunc,daemon=True).start()

        Thread(target=self.frameAnalyzeThreadFunc_video,daemon=True).start()



    def setupUI(self):
        #大小，图标，界面标题
        self.resize(1100,600)
        self.setWindowTitle("YOLO识别系统-wushuyue")
        self.setWindowIcon(QtGui.QIcon("images\wushuyue.gif"))

        #central Widget
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)

        mainLayout = QtWidgets.QVBoxLayout(self.centralWidget)


        #显示部分(上部分)
        self.topLayout = QtWidgets.QHBoxLayout()
        self.label_ori = QtWidgets.QLabel(self)
        self.label_result = QtWidgets.QLabel(self)

        self.label_ori.setMinimumSize(520,400)
        self.label_result.setMinimumSize(520,400)
        self.label_ori.setStyleSheet("border:1px solid black")
        self.label_result.setStyleSheet("border:1px solid black")

        self.topLayout.addWidget(self.label_ori)
        self.topLayout.addWidget(self.label_result)

        mainLayout.addLayout(self.topLayout)

        #结果与控制部分（下）

        groupBox_left = QtWidgets.QGroupBox(self)
        groupBox_right = QtWidgets.QGroupBox(self)



        self.bottomLayout_control = QtWidgets.QHBoxLayout(groupBox_left)
        self.bottomLayout = QtWidgets.QVBoxLayout(groupBox_right)

        #制作控制台
        #选择模型大小功能（目前只支持n和s大小）

        self.white_add1 = QtWidgets.QLabel(self)
        self.white_add2 = QtWidgets.QLabel(self)

        self.model_label =QtWidgets.QLabel(self)
        self.model_label.setText("模型大小：")
        self.model_label.setStyleSheet("border:0px")
        self.model_label.setStyleSheet("font-size:20px")
        self.bottomLayout_control.addWidget(self.model_label)

        self.model_comba = QtWidgets.QComboBox(self)
        self.model_comba.addItems(["yolov10n.pt","yolov10s.pt"])
        self.model_comba.setStyleSheet("font-size:15px")
        self.model_comba.currentIndexChanged.connect(self.on_model_changed)
        self.bottomLayout_control.addWidget(self.model_comba)
        
        self.bottomLayout_control.addWidget(self.white_add1)


        #选择模式功能（目前可以支持目标检测和实例分割）
        self.mode_label =QtWidgets.QLabel(self)
        self.mode_label.setText("选择模式：")
        self.mode_label.setStyleSheet("border:0px")
        self.mode_label.setStyleSheet("font-size:20px")
        self.bottomLayout_control.addWidget(self.mode_label)

        self.mode_comba = QtWidgets.QComboBox(self)
        self.mode_comba.addItems(["Object Detection","Segmentation"])
        self.mode_comba.setStyleSheet("font-size:15px")
        self.mode_comba.currentIndexChanged.connect(self.on_mode_changed)
        self.bottomLayout_control.addWidget(self.mode_comba)

        self.bottomLayout_control.addWidget(self.white_add2)

        mainLayout.addWidget(groupBox_left)
        
        #控制按钮
        self.btnLayout = QtWidgets.QHBoxLayout()

        button_style = '''
        QPushButton {  
            border:1px solid gray;     /* 边框颜色和粗细 */  
            border-radius: 8px;       /* 边框圆角 */  
            padding: 5px 10px;         /* 内边距 */  
            background-color: #f0f0f0; /* 背景颜色 */  
            color: black;             /* 文字颜色 */  
            font-size: 20px;          /* 字体大小 */  
        } 
        QPushButton:hover{
            background-color : #e0e0e0;
        }
        '''

        self.video_btn = QtWidgets.QPushButton("🎞️视频文件")
        self.video_btn.clicked.connect(self.getVideo)
        self.video_btn.setStyleSheet(button_style)

        self.cam_btn = QtWidgets.QPushButton("📹摄像头")
        self.cam_btn.clicked.connect(self.startCamera)
        self.cam_btn.clicked.connect(self.ChangeFlag_camera)
        self.cam_btn.setStyleSheet(button_style)

        self.img_btn = QtWidgets.QPushButton("📷图片")
        self.img_btn.clicked.connect(self.get_image)
        self.img_btn.setStyleSheet(button_style)


        self.stop_btn = QtWidgets.QPushButton("🛑停止")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setStyleSheet(button_style)


        self.btnLayout.addWidget(self.cam_btn)
        self.btnLayout.addWidget(self.video_btn)
        self.btnLayout.addWidget(self.img_btn)
        self.btnLayout.addWidget(self.stop_btn)

        self.bottomLayout.setSpacing(40)

        mainLayout.addWidget(groupBox_right)


    def ChangeFlag_camera(self):
        self.flag_camera = True


    def getVideo(self):
        file_dialog = QtWidgets.QFileDialog()
        self.video_path = file_dialog.getOpenFileName(
            self,
            "选择你要上传的视频",
            "D:\\",
            "视频类型(*.avi *.mp4 *.mov *.flv)"      #AVI、MP4、MOV
        )[0]
        if self.video_path == "":
            return
        self.flag_video = True
        self.cap_video = cv2.VideoCapture(self.video_path)
        if not self.cap_video.isOpened():
            print("视频打开失败，请重试")
            return
        if self.timer_video.isActive() == False:
            self.timer_video.start(27)

    
    def show_video(self):
        ret,frame = self.cap_video.read()

        if not ret:
            return
        
        frame = cv2.resize(frame,(520,400))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(frame.data,frame.shape[1],frame.shape[0],
                     QtGui.QImage.Format.Format_RGB888)#pyqt5与pyqt6中的枚举值机制改了
        
        self.label_ori.setPixmap(QtGui.QPixmap.fromImage(qImage))

        if not self.frameToanalyze_video:
            self.frameToanalyze_video.append(frame)


    def startCamera(self):
        # 打开摄像头
        self.cap_camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        if not self.cap_camera.isOpened():
            print("摄像头打开失败，请检查设备")
            return
        #如果没启动，则启动（防止连续点击按钮造成启动多个定时器）
        if self.timer_camera.isActive() == False:
            self.timer_camera.start(27)

    def show_camera(self):
         # 读取视频流
        ret,frame = self.cap_camera.read()

        if not ret:
            return
        
        #对从opencv中获取的数据重新设参
        frame = cv2.resize(frame,(520,400))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(frame.data,frame.shape[1],frame.shape[0],
                     QtGui.QImage.Format.Format_RGB888)#pyqt5与pyqt6中的枚举值机制改了
        
        self.label_ori.setPixmap(QtGui.QPixmap.fromImage(qImage))

        if not self.frameToanalyze_camera:
            self.frameToanalyze_camera.append(frame)

    def frameAnalyzeThreadFunc(self):
        
        while True:
            if not self.frameToanalyze_camera:
                time.sleep(0.01)
                continue
            
            frame = self.frameToanalyze_camera.pop(0)

            results = self.model(frame)[0]

            if self.flag_mode_det:
                img = results.plot(line_width = 1)
            else:
                img = results.plot(line_width = 0)

            qImage = QtGui.QImage(img.data,img.shape[1],img.shape[0],
                             QtGui.QImage.Format.Format_RGB888)
            
            self.label_result.setPixmap(QtGui.QPixmap.fromImage(qImage))

            time.sleep(0.02)

    def frameAnalyzeThreadFunc_video(self):
        
        while True:
            if not self.frameToanalyze_video:
                time.sleep(0.01)
                continue
            
            frame = self.frameToanalyze_video.pop(0)

            results = self.model(frame)[0]

            if self.flag_mode_det:
                img = results.plot(line_width = 1)
            else:
                img = results.plot(line_width = 0)
            qImage = QtGui.QImage(img.data,img.shape[1],img.shape[0],
                             QtGui.QImage.Format.Format_RGB888)
            
            self.label_result.setPixmap(QtGui.QPixmap.fromImage(qImage))

            time.sleep(0.03)

    
    def get_image(self):
        file_dialog = QtWidgets.QFileDialog()
        self.image_path = file_dialog.getOpenFileName(
            self,
            "选择你要上传的图片", # 标题
            'D:\\' ,       # 起始目录
            "图片类型 (*.png *.jpg *.bmp)" # 选择类型过滤项，过滤内容在括号中
        )[0]
        if self.image_path == "":
            return
        self.flag_image = True
        self.show_image()

    def show_image(self):
        frame = cv2.imread(self.image_path)
        frame = cv2.resize(frame,(520,400))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(frame.data,frame.shape[1],frame.shape[0],
                     QtGui.QImage.Format.Format_RGB888)
        self.label_ori.setPixmap(QtGui.QPixmap.fromImage(qImage))

        result = self.model(frame)[0]

        if self.flag_mode_det:
            img =result.plot(line_width = 2)
        else:
            img = result.plot(line_width = 0)
        qImage = QtGui.QImage(img.data,img.shape[1],img.shape[0],
                             QtGui.QImage.Format.Format_RGB888)
            
        self.label_result.setPixmap(QtGui.QPixmap.fromImage(qImage))


        
    def stop(self):
        if self.flag_video:
            self.timer_video.stop()
            self.flag_video = False
            time.sleep(0.36)
            self.label_result.clear()

        if self.flag_camera:
            self.timer_camera.stop()    #关闭定时器
            self.flag_camera = False
            time.sleep(0.35)
            self.label_result.clear()

        self.flag_image = False
        self.label_ori.clear()
        self.label_result.clear()   #清空视频显示区域

    
    def on_model_changed(self,index):
        detection_paths=["model\yolov8n.pt","model\yolov8s.pt"]
        segmentation_paths=["model\yolov8n-seg.pt","model\yolov8s-seg.pt"]
        if self.flag_mode_det:
            model_path = detection_paths[index]
            self.model = YOLO(model_path)
        else:
            model_path = segmentation_paths[index]
            self.model = YOLO(model_path)
        if self.flag_image:
            self.show_image()

    def on_mode_changed(self,index):
        detection_paths=["model\yolov8n.pt","model\yolov8s.pt"]
        segmentation_paths=["model\yolov8n-seg.pt","model\yolov8s-seg.pt"]
        if index == 0:
            self.flag_mode_det = True
            mode_path = detection_paths[self.model_scale]
            self.model = YOLO(mode_path)
        elif index == 1:
            self.flag_mode_det = False
            mode_path = segmentation_paths[self.model_scale]
            self.model = YOLO(mode_path)
        if self.flag_image:
            self.show_image()
                

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Main_Window()
    window.show()
    sys.exit(app.exec())

