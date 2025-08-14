"""Animation Creation with FiveThirtyEight Style"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Optional, Tuple, Any
from tqdm import tqdm

plt.style.use('fivethirtyeight')


class Animator:
    """Create FiveThirtyEight-style animations for NLSE solutions."""
    
    @staticmethod
    def create_gif(x: np.ndarray, t: np.ndarray, psi: np.ndarray,
                  filename: str, output_dir: str = "../outputs",
                  title: str = "NLSE Evolution", xlim: Optional[Tuple[float, float]] = None,
                  fps: int = 20, dpi: int = 100, verbose: bool = True,
                  logger: Optional[Any] = None) -> None:
        """Create animated GIF of wave function evolution in FiveThirtyEight style."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        if xlim is None:
            xlim = (x[0], x[-1])
        
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#f0f0f0')
        
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#f8f8f8')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            ax.tick_params(colors='#666666')
        
        line1, = ax1.plot(x, np.abs(psi[0]), linewidth=2.5, color='#008fd5')
        line2, = ax2.plot(x, np.real(psi[0]), linewidth=2.5, color='#fc4f30', label='Re(ψ)', alpha=0.8)
        line3, = ax2.plot(x, np.imag(psi[0]), linewidth=2.5, color='#6d904f', label='Im(ψ)', alpha=0.8)
        
        ax1.set_xlim(xlim)
        ax1.set_ylim([0, np.max(np.abs(psi)) * 1.1])
        ax1.set_xlabel('Position x', fontsize=11, color='#333333')
        ax1.set_ylabel('|ψ|', fontsize=11, color='#333333')
        ax1.grid(True, alpha=0.4, color='#cccccc', linewidth=0.5)
        
        ax2.set_xlim(xlim)
        y_max = max(np.max(np.abs(np.real(psi))), np.max(np.abs(np.imag(psi))))
        ax2.set_ylim([-y_max * 1.1, y_max * 1.1])
        ax2.set_xlabel('Position x', fontsize=11, color='#333333')
        ax2.set_ylabel('ψ', fontsize=11, color='#333333')
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.4, color='#cccccc', linewidth=0.5)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', color='#333333', y=0.98)
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                            fontsize=10, color='#666666',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.text(0.99, 0.01, 'Samudera Sains Teknologi Ltd.', 
                ha='right', va='bottom', fontsize=8, color='#999999', alpha=0.7)
        
        def animate(frame):
            line1.set_ydata(np.abs(psi[frame]))
            line2.set_ydata(np.real(psi[frame]))
            line3.set_ydata(np.imag(psi[frame]))
            time_text.set_text(f't = {t[frame]:.2f}')
            return line1, line2, line3, time_text
        
        anim = FuncAnimation(fig, animate, frames=len(t), interval=1000/fps, blit=True)
        anim.save(filepath, writer='pillow', fps=fps, dpi=dpi)
        plt.close()
