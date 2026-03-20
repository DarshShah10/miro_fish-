"""
数据模型模块
"""

from .task import TaskManager, TaskStatus
from .project import Project, ProjectStatus, ProjectManager

# Emergency Response Models
from .emergency_case import (
    EmergencyCase,
    DistressSignal,
    EmergencySeverity,
    EmergencyType,
    Location,
    PatientInfo,
    Intervention,
    InterventionAction
)

from .response_resource import (
    ResourceRegistry,
    Hospital,
    Ambulance,
    MedicalStaff,
    BloodBank,
    TransportRoute,
    ResourceLocation,
    HospitalLevel,
    StaffSpecialization,
    ResourceStatus
)

from .response_action import (
    ResponseActionType,
    AgentType,
    ActionMessage,
    AgentState,
    ActionLog,
    AgentStates
)

__all__ = [
    # Base models
    'TaskManager', 'TaskStatus',
    'Project', 'ProjectStatus', 'ProjectManager',
    # Emergency case models
    'EmergencyCase', 'DistressSignal', 'EmergencySeverity',
    'EmergencyType', 'Location', 'PatientInfo',
    'Intervention', 'InterventionAction',
    # Response resource models
    'ResourceRegistry', 'Hospital', 'Ambulance',
    'MedicalStaff', 'BloodBank', 'TransportRoute',
    'ResourceLocation', 'HospitalLevel',
    'StaffSpecialization', 'ResourceStatus',
    # Response action models
    'ResponseActionType', 'AgentType',
    'ActionMessage', 'AgentState', 'ActionLog', 'AgentStates'
]

