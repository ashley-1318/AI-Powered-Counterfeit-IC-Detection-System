"""
CircuitCheck Arduino Integration Module

This module handles communication with the Arduino test jig for
performing electrical measurements on electronic components.

Features:
- Serial communication with Arduino
- Test command execution
- Data parsing and validation
- Error handling and recovery

Author: CircuitCheck Team
Version: 1.0
"""

import serial
import json
import time
import logging
import threading
from typing import Dict, Optional, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArduinoController:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        """
        Initialize Arduino controller
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Communication speed
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.is_connected = False
        self.test_in_progress = False
        self.timeout = 30  # 30 second timeout for tests
        
    def connect(self) -> bool:
        """
        Connect to Arduino
        
        Returns:
            bool: Connection success status
        """
        try:
            if self.port is None:
                self.port = self._auto_detect_port()
                
            if self.port is None:
                logger.error("No Arduino port detected")
                return False
            
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2
            )
            
            # Wait for Arduino to initialize
            time.sleep(2)
            
            # Test connection with ping
            if self._ping_arduino():
                self.is_connected = True
                logger.info(f"Connected to Arduino on {self.port}")
                return True
            else:
                logger.error("Arduino not responding to ping")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.serial_connection:
            self.serial_connection.close()
            self.is_connected = False
            logger.info("Disconnected from Arduino")
    
    def _auto_detect_port(self) -> Optional[str]:
        """
        Auto-detect Arduino port
        
        Returns:
            str: Detected port or None
        """
        import serial.tools.list_ports
        
        # Common Arduino port patterns
        arduino_patterns = [
            'Arduino',
            'CH340',
            'CP210x',
            'FT232',
            'ttyUSB',
            'ttyACM'
        ]
        
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            port_desc = port.description.upper()
            port_name = port.device.upper()
            
            for pattern in arduino_patterns:
                if pattern.upper() in port_desc or pattern.upper() in port_name:
                    logger.info(f"Detected Arduino on port: {port.device}")
                    return port.device
        
        logger.warning("Could not auto-detect Arduino port")
        return None
    
    def _ping_arduino(self) -> bool:
        """
        Test Arduino connection
        
        Returns:
            bool: Arduino responding
        """
        try:
            self._send_command("PING")
            response = self._read_response(timeout=5)
            return response == "PONG"
        except:
            return False
    
    def _send_command(self, command: str):
        """Send command to Arduino"""
        if not self.is_connected or self.serial_connection is None:
            raise Exception("Arduino not connected")
        
        command_bytes = (command + '\n').encode('utf-8')
        self.serial_connection.write(command_bytes)
        logger.debug(f"Sent command: {command}")
    
    def _read_response(self, timeout: int = 10) -> str:
        """
        Read response from Arduino
        
        Args:
            timeout: Read timeout in seconds
            
        Returns:
            str: Response string
        """
        if not self.is_connected or self.serial_connection is None:
            raise Exception("Arduino not connected")
        
        start_time = time.time()
        response = ""
        
        while time.time() - start_time < timeout:
            if self.serial_connection.in_waiting > 0:
                line = self.serial_connection.readline().decode('utf-8').strip()
                if line:
                    response = line
                    logger.debug(f"Received: {line}")
                    break
            time.sleep(0.1)
        
        if not response:
            raise Exception(f"No response from Arduino within {timeout} seconds")
        
        return response
    
    def get_status(self) -> Dict:
        """
        Get Arduino status
        
        Returns:
            dict: Status information
        """
        try:
            self._send_command("GET_STATUS")
            
            # Wait for status line
            while True:
                response = self._read_response()
                if response.startswith("STATUS:"):
                    # Read JSON data
                    json_response = self._read_response()
                    return json.loads(json_response)
                    
        except Exception as e:
            logger.error(f"Failed to get Arduino status: {e}")
            return {"error": str(e)}
    
    def perform_electrical_test(self) -> Dict:
        """
        Perform electrical test on connected component
        
        Returns:
            dict: Test results
        """
        if not self.is_connected:
            raise Exception("Arduino not connected")
        
        if self.test_in_progress:
            raise Exception("Test already in progress")
        
        try:
            self.test_in_progress = True
            logger.info("Starting electrical test...")
            
            # Send test command
            self._send_command("START_TEST")
            
            # Wait for test to complete
            start_time = time.time()
            
            while time.time() - start_time < self.timeout:
                response = self._read_response(timeout=5)
                
                if response.startswith("RESULT:"):
                    # Read JSON result
                    json_response = self._read_response(timeout=10)
                    result = json.loads(json_response)
                    
                    # Wait for completion message
                    completion_msg = self._read_response(timeout=5)
                    logger.info(f"Test completed: {completion_msg}")
                    
                    return result
                    
                elif response.startswith("ERROR:"):
                    raise Exception(f"Arduino error: {response}")
                    
                elif response.startswith("STATUS:"):
                    logger.info(f"Test status: {response}")
                    continue
            
            raise Exception("Test timeout")
            
        except Exception as e:
            logger.error(f"Electrical test failed: {e}")
            return {"error": str(e), "status": "failed"}
            
        finally:
            self.test_in_progress = False
    
    def reset_system(self) -> bool:
        """
        Reset Arduino system
        
        Returns:
            bool: Reset success
        """
        try:
            self._send_command("RESET")
            response = self._read_response()
            
            if "reset completed" in response.lower():
                logger.info("Arduino system reset")
                return True
            else:
                logger.error(f"Reset failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reset Arduino: {e}")
            return False

def detect_arduino_ports() -> List[str]:
    """
    Detect available Arduino ports
    
    Returns:
        list: List of potential Arduino ports
    """
    import serial.tools.list_ports
    
    arduino_ports = []
    arduino_patterns = [
        'Arduino',
        'CH340',
        'CP210x',
        'FT232',
        'ttyUSB',
        'ttyACM'
    ]
    
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        port_desc = port.description.upper()
        port_name = port.device.upper()
        
        for pattern in arduino_patterns:
            if pattern.upper() in port_desc or pattern.upper() in port_name:
                arduino_ports.append(port.device)
                break
    
    return arduino_ports

def test_arduino_connection(port: Optional[str] = None) -> bool:
    """
    Test Arduino connection
    
    Args:
        port: Serial port to test
        
    Returns:
        bool: Connection test result
    """
    controller = ArduinoController(port)
    
    if controller.connect():
        status = controller.get_status()
        print(f"Arduino Status: {json.dumps(status, indent=2)}")
        
        controller.disconnect()
        return True
    else:
        return False

# Example usage and testing
if __name__ == "__main__":
    print("Testing Arduino connection...")
    
    # Detect available ports
    ports = detect_arduino_ports()
    print(f"Detected Arduino ports: {ports}")
    
    if not ports:
        print("No Arduino ports detected!")
        exit(1)
    
    # Test connection to first port
    controller = ArduinoController(ports[0])
    
    if controller.connect():
        print("Connection successful!")
        
        # Get status
        status = controller.get_status()
        print(f"Status: {json.dumps(status, indent=2)}")
        
        # Perform test
        print("Starting electrical test...")
        result = controller.perform_electrical_test()
        print(f"Test result: {json.dumps(result, indent=2)}")
        
        controller.disconnect()
    else:
        print("Connection failed!")