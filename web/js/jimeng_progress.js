import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "ComfyUI.Jimeng.ProgressBar",

    async setup() {
        api.addEventListener("progress", ({ detail }) => {
            const { value, max, node } = detail;
            const graphNode = app.graph.getNodeById(node);
            
            if (graphNode && graphNode.comfyClass.startsWith("Jimeng")) {
                const ratio = value / max;
                graphNode.jimeng_progress_ratio = ratio;
                graphNode.jimeng_progress_text = `${value}s / ${max}s`;
                
                app.graph.setDirtyCanvas(true, false);
            }
        });

        api.addEventListener("jimeng_fake_progress", ({ detail }) => {
            const { max, node } = detail;
            const graphNode = app.graph.getNodeById(node);

            if (graphNode && graphNode.comfyClass.startsWith("Jimeng")) {
                graphNode.jimeng_progress_mode = "auto";
                graphNode.jimeng_progress_startTime = performance.now();
                graphNode.jimeng_progress_duration = Math.max(max, 1) * 1000;
                graphNode.jimeng_progress_ratio = 0;
                graphNode.jimeng_progress_text = "";

                app.graph.setDirtyCanvas(true, false);
            }
        });

        api.addEventListener("executed", ({ detail }) => {
             const graphNode = app.graph.getNodeById(detail.node);
             if (graphNode && graphNode.comfyClass.startsWith("Jimeng")) {
                 graphNode.jimeng_progress_ratio = 0;
                 graphNode.jimeng_progress_text = "";
                 graphNode.jimeng_progress_mode = null;
                 graphNode.jimeng_progress_startTime = 0;
                 graphNode.jimeng_progress_duration = 0;
                 app.graph.setDirtyCanvas(true, false);
             }
        });
    },

    nodeCreated(node) {
        if (node.comfyClass.startsWith("Jimeng")) {
            const origOnDrawForeground = node.onDrawForeground;
            
            node.onDrawForeground = function(ctx) {
                if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);

                if (this.jimeng_progress_mode === "auto" && this.jimeng_progress_startTime) {
                    const now = performance.now();
                    const duration = this.jimeng_progress_duration || 1000;
                    const elapsed = now - this.jimeng_progress_startTime;
                    const ratio = elapsed / duration;

                    if (ratio >= 1) {
                        this.jimeng_progress_mode = null;
                        this.jimeng_progress_startTime = 0;
                        this.jimeng_progress_duration = 0;
                        this.jimeng_progress_ratio = 0;
                        this.jimeng_progress_text = "";
                    } else if (ratio > 0) {
                        this.jimeng_progress_ratio = ratio;
                        this.jimeng_progress_text = `${Math.round(elapsed / 1000)}s / ${Math.round(duration / 1000)}s`;
                        app.graph.setDirtyCanvas(true, false);
                    }
                }

                if (this.jimeng_progress_ratio > 0 && this.jimeng_progress_ratio < 1) {
                    const w = this.size[0];
                    // 进度条高度
                    const titleHeight = 0;
                    const barHeight = 4;
                    const barY = titleHeight;
                    
                    const textX = w - 10;
                    // 文字高度
                    const textY = -14;

                    ctx.save();

                    // 进度条背景槽
                    ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
                    ctx.fillRect(0, barY, w, barHeight);

                    // 进度条颜色
                    ctx.fillStyle = "#4caf50"; 
                    ctx.beginPath();
                    if (this.jimeng_progress_ratio > 0.99) {
                        ctx.rect(0, barY, w * this.jimeng_progress_ratio, barHeight); 
                    } else {
                        ctx.roundRect(0, barY, w * this.jimeng_progress_ratio, barHeight, [0, 4, 4, 0]);
                    }
                    ctx.fill();
                    
                    // 绘制文字
                    if (this.jimeng_progress_text) {
                        ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
                        ctx.font = "12px Arial";
                        ctx.textAlign = "right";
                        ctx.textBaseline = "middle";
                        ctx.fillText(this.jimeng_progress_text, textX, textY);
                    }
                    
                    ctx.restore();
                }
            };
        }
    }
});
