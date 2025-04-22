"""
白酒品质检测系统组件包

此包包含了白酒品质检测系统的所有UI组件。
"""

# 导出所有组件类，以便可以直接从components包导入
from components.login_window import LoginWindow
from components.register_window import RegisterWindow
from components.main_application import MainApplication
from components.user_info_frame import UserInfoFrame
from components.analysis_frame import AnalysisFrame
from components.classification_frame import ClassificationFrame
from components.history_frame import HistoryFrame
from components.settings_frame import SettingsFrame
from components.user_management_frame import UserManagementFrame

# 版本信息
__version__ = "1.0.0" 