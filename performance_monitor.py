#!/usr/bin/env python3
"""
Real-time Performance Monitor for TensorFlow GPU

Monitors GPU utilization, memory usage, temperature, and TensorFlow performance
metrics in real-time with web dashboard support.
"""

import time
import psutil
import threading
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

import tensorflow as tf
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from rich.text import Text

console = Console()

@dataclass
class GPUMetrics:
    """GPU metrics data structure."""
    timestamp: str
    gpu_id: int
    name: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature_c: float
    power_draw_w: float
    power_limit_w: float
    fan_speed_percent: float

@dataclass
class SystemMetrics:
    """System metrics data structure."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    network_sent_mb: float
    network_recv_mb: float

@dataclass
class TensorFlowMetrics:
    """TensorFlow specific metrics."""
    timestamp: str
    tf_version: str
    gpu_available: bool
    mixed_precision_enabled: bool
    xla_enabled: bool
    current_operation: Optional[str] = None
    operations_per_sec: Optional[float] = None

class PerformanceMonitor:
    """Real-time performance monitoring for TensorFlow GPU systems."""
    
    def __init__(self, log_file: str = "performance_log.jsonl"):
        self.log_file = Path(log_file)
        self.running = False
        self.metrics_history: List[Dict] = []
        self.max_history = 1000
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_initialized = True
                self.gpu_count = nvml.nvmlDeviceGetCount()
            except Exception as e:
                console.print(f"Failed to initialize NVML: {e}", style="red")
                self.nvml_initialized = False
                self.gpu_count = 0
        else:
            self.nvml_initialized = False
            self.gpu_count = 0
        
        # TensorFlow setup
        self.tf_metrics = TensorFlowMetrics(
            timestamp=datetime.now().isoformat(),
            tf_version=tf.__version__,
            gpu_available=len(tf.config.list_physical_devices('GPU')) > 0,
            mixed_precision_enabled=tf.keras.mixed_precision.global_policy().name != 'float32',
            xla_enabled=tf.config.optimizer.get_jit() is not None
        )
        
        # Initialize baseline network stats
        self.baseline_net = psutil.net_io_counters()
        self.start_time = time.time()
    
    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """Get GPU metrics using NVML."""
        metrics = []
        
        if not self.nvml_initialized:
            return metrics
        
        try:
            for gpu_id in range(self.gpu_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Basic info
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature
                try:
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = 0
                
                # Power
                try:
                    power_draw = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_limit = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                except:
                    power_draw = power_limit = 0
                
                # Fan speed
                try:
                    fan_speed = nvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    fan_speed = 0
                
                metrics.append(GPUMetrics(
                    timestamp=datetime.now().isoformat(),
                    gpu_id=gpu_id,
                    name=name,
                    utilization_percent=util.gpu,
                    memory_used_mb=mem_info.used / 1024 / 1024,
                    memory_total_mb=mem_info.total / 1024 / 1024,
                    memory_percent=(mem_info.used / mem_info.total) * 100,
                    temperature_c=temp,
                    power_draw_w=power_draw,
                    power_limit_w=power_limit,
                    fan_speed_percent=fan_speed
                ))
        except Exception as e:
            console.print(f"Error getting GPU metrics: {e}", style="red")
        
        return metrics
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get system metrics using psutil."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network
        current_net = psutil.net_io_counters()
        elapsed = time.time() - self.start_time
        
        if elapsed > 0:
            net_sent_mb = (current_net.bytes_sent - self.baseline_net.bytes_sent) / 1024 / 1024
            net_recv_mb = (current_net.bytes_recv - self.baseline_net.bytes_recv) / 1024 / 1024
        else:
            net_sent_mb = net_recv_mb = 0
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / 1024 / 1024 / 1024,
            memory_total_gb=memory.total / 1024 / 1024 / 1024,
            disk_percent=disk.percent,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb
        )
    
    def create_dashboard(self) -> Layout:
        """Create rich dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="gpu_info"),
            Layout(name="system_info")
        )
        
        layout["right"].split_column(
            Layout(name="tensorflow_info"),
            Layout(name="performance_chart")
        )
        
        return layout
    
    def update_dashboard(self, layout: Layout):
        """Update dashboard with current metrics."""
        # Header
        header_text = Text("TensorFlow GPU Performance Monitor", style="bold magenta")
        layout["header"].update(Panel(header_text, title="Real-time Monitoring"))
        
        # GPU metrics
        gpu_metrics = self.get_gpu_metrics()
        if gpu_metrics:
            gpu_table = Table(title="GPU Status")
            gpu_table.add_column("GPU", style="cyan")
            gpu_table.add_column("Util %", style="green")
            gpu_table.add_column("Memory", style="yellow")
            gpu_table.add_column("Temp °C", style="red")
            gpu_table.add_column("Power W", style="blue")
            
            for gpu in gpu_metrics:
                memory_str = f"{gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f} MB ({gpu.memory_percent:.1f}%)"
                gpu_table.add_row(
                    f"GPU {gpu.gpu_id}",
                    f"{gpu.utilization_percent:.1f}%",
                    memory_str,
                    f"{gpu.temperature_c:.0f}°C",
                    f"{gpu.power_draw_w:.1f}/{gpu.power_limit_w:.1f}W"
                )
            
            layout["gpu_info"].update(Panel(gpu_table))
        else:
            layout["gpu_info"].update(Panel("No GPU metrics available", title="GPU Status"))
        
        # System metrics
        sys_metrics = self.get_system_metrics()
        sys_table = Table(title="System Status")
        sys_table.add_column("Metric", style="cyan")
        sys_table.add_column("Value", style="white")
        
        sys_table.add_row("CPU Usage", f"{sys_metrics.cpu_percent:.1f}%")
        sys_table.add_row("Memory", f"{sys_metrics.memory_used_gb:.1f}/{sys_metrics.memory_total_gb:.1f} GB ({sys_metrics.memory_percent:.1f}%)")
        sys_table.add_row("Disk Usage", f"{sys_metrics.disk_percent:.1f}%")
        sys_table.add_row("Network Sent", f"{sys_metrics.network_sent_mb:.1f} MB")
        sys_table.add_row("Network Recv", f"{sys_metrics.network_recv_mb:.1f} MB")
        
        layout["system_info"].update(Panel(sys_table))
        
        # TensorFlow info
        tf_table = Table(title="TensorFlow Status")
        tf_table.add_column("Setting", style="cyan")
        tf_table.add_column("Value", style="white")
        
        tf_table.add_row("Version", self.tf_metrics.tf_version)
        tf_table.add_row("GPU Available", "✅" if self.tf_metrics.gpu_available else "❌")
        tf_table.add_row("Mixed Precision", "✅" if self.tf_metrics.mixed_precision_enabled else "❌")
        tf_table.add_row("XLA Enabled", "✅" if self.tf_metrics.xla_enabled else "❌")
        
        layout["tensorflow_info"].update(Panel(tf_table))
        
        # Footer with instructions
        footer_text = Text("Press Ctrl+C to stop monitoring | Logs saved to performance_log.jsonl", style="dim")
        layout["footer"].update(Panel(footer_text))
    
    def log_metrics(self, gpu_metrics: List[GPUMetrics], sys_metrics: SystemMetrics):
        """Log metrics to JSON file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "gpu_metrics": [asdict(gpu) for gpu in gpu_metrics],
            "system_metrics": asdict(sys_metrics),
            "tensorflow_metrics": asdict(self.tf_metrics)
        }
        
        # Add to history
        self.metrics_history.append(log_entry)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def start_monitoring(self, interval: float = 1.0):
        """Start real-time monitoring with dashboard."""
        self.running = True
        layout = self.create_dashboard()
        
        console.print("Starting performance monitoring...", style="green")
        console.print(f"Logging to: {self.log_file}", style="blue")
        
        try:
            with Live(layout, refresh_per_second=1, screen=True):
                while self.running:
                    # Get metrics
                    gpu_metrics = self.get_gpu_metrics()
                    sys_metrics = self.get_system_metrics()
                    
                    # Update dashboard
                    self.update_dashboard(layout)
                    
                    # Log metrics
                    self.log_metrics(gpu_metrics, sys_metrics)
                    
                    time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\nStopping monitoring...", style="yellow")
        finally:
            self.running = False
    
    def generate_report(self) -> str:
        """Generate performance report from logged data."""
        if not self.metrics_history:
            return "No performance data available."
        
        # Analyze the data
        gpu_utilizations = []
        memory_usages = []
        temperatures = []
        
        for entry in self.metrics_history:
            for gpu in entry.get('gpu_metrics', []):
                gpu_utilizations.append(gpu['utilization_percent'])
                memory_usages.append(gpu['memory_percent'])
                temperatures.append(gpu['temperature_c'])
        
        if gpu_utilizations:
            avg_util = sum(gpu_utilizations) / len(gpu_utilizations)
            max_util = max(gpu_utilizations)
            avg_mem = sum(memory_usages) / len(memory_usages)
            max_temp = max(temperatures)
            
            report = f"""
Performance Report
================
Monitoring Duration: {len(self.metrics_history)} samples
GPU Utilization: {avg_util:.1f}% avg, {max_util:.1f}% max
Memory Usage: {avg_mem:.1f}% avg
Max Temperature: {max_temp:.1f}°C

TensorFlow Configuration:
- Version: {self.tf_metrics.tf_version}
- GPU Available: {self.tf_metrics.gpu_available}
- Mixed Precision: {self.tf_metrics.mixed_precision_enabled}
- XLA Enabled: {self.tf_metrics.xla_enabled}
"""
        else:
            report = "No GPU metrics collected."
        
        return report

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorFlow GPU Performance Monitor")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds")
    parser.add_argument("--log-file", default="performance_log.jsonl", help="Log file path")
    parser.add_argument("--report", action="store_true", help="Generate report from existing log file")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(log_file=args.log_file)
    
    if args.report:
        # Load existing data and generate report
        if monitor.log_file.exists():
            with open(monitor.log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        monitor.metrics_history.append(entry)
                    except:
                        pass
        
        report = monitor.generate_report()
        console.print(report)
    else:
        # Start real-time monitoring
        monitor.start_monitoring(interval=args.interval)

if __name__ == "__main__":
    main()