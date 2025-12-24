import { app } from "/scripts/app.js";

/**
 * 根据名称查找组件
 */
function findWidgetByName(node, name) {
    if (!node.widgets) return null;
    return node.widgets.find((w) => w.name === name);
}

/**
 * 切换组件可见性
 */
function toggleWidget(node, widget, show) {
    if (!widget) return;

    // 备份原始类型和尺寸计算函数
    if (!widget.origType && widget.type !== "hidden") {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
    }

    if (!widget.origType && widget.type === "hidden") {
        return;
    }

    if (show) {
        widget.type = widget.origType;
        widget.computeSize = widget.origComputeSize;
    } else {
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    }
}

/**
 * 更新节点高度
 */
function updateNodeHeight(node) {
    if (node.flags && node.flags.collapsed) return;

    const size = node.computeSize();
    node.setSize([node.size[0], size[1]]);
    app.graph.setDirtyCanvas(true, true);
}

// 添加了 'auto_duration'
const TARGET_WIDGETS = ['size', 'enable_group_generation', 'generation_count', 'enable_timeout_setting', 'enable_random_seed', 'auto_duration'];

/**
 * 处理组件联动逻辑
 */
function widgetLogic(node, widget) {
    // 图像节点逻辑
    if (node.comfyClass === "JimengSeedream3" || node.comfyClass === "JimengSeedream4") {
        if (widget.name === 'size') {
            const isCustom = widget.value === "Custom";
            const widthWidget = findWidgetByName(node, 'width');
            const heightWidget = findWidgetByName(node, 'height');

            if (widthWidget) toggleWidget(node, widthWidget, isCustom);
            if (heightWidget) toggleWidget(node, heightWidget, isCustom);
        }
    }

    // 分组生成逻辑
    if (node.comfyClass === "JimengSeedream4") {
        if (widget.name === 'enable_group_generation') {
            const isGroupMode = widget.value === true;
            const maxImagesWidget = findWidgetByName(node, 'max_images');

            if (maxImagesWidget) toggleWidget(node, maxImagesWidget, isGroupMode);
        }
    }

    // 视频节点逻辑：加入了 JimengSeedance1_5
    if (node.comfyClass === "JimengSeedance1" || node.comfyClass === "JimengReferenceImage2Video" || node.comfyClass === "JimengSeedance1_5") {

        // 1.5 特有逻辑：智能时长
        if (widget.name === 'auto_duration') {
            const isAuto = widget.value === true;
            const durationWidget = findWidgetByName(node, 'duration');

            // 启用智能时长时,隐藏 duration (show = !isAuto)
            if (durationWidget) toggleWidget(node, durationWidget, !isAuto);
        }

        // 通用逻辑：生成数量 > 1 显示批量保存选项
        if (widget.name === 'generation_count') {
            const isBatch = widget.value > 1;

            const batchPathWidget = findWidgetByName(node, 'filename_prefix');
            const saveLastFrameWidget = findWidgetByName(node, 'save_last_frame_batch');

            if (batchPathWidget) toggleWidget(node, batchPathWidget, isBatch);
            if (saveLastFrameWidget) toggleWidget(node, saveLastFrameWidget, isBatch);
        }

        // 通用逻辑：超时设置
        if (widget.name === 'enable_timeout_setting') {
            const isTimeoutEnabled = widget.value === true;
            const timeoutWidget = findWidgetByName(node, 'timeout_seconds');

            if (timeoutWidget) toggleWidget(node, timeoutWidget, isTimeoutEnabled);
        }

        // 通用逻辑：随机种子
        if (widget.name === 'enable_random_seed') {
            const useRandom = widget.value === true;
            const showSeedControls = !useRandom;

            const seedWidget = findWidgetByName(node, 'seed');
            const controlWidget = findWidgetByName(node, 'control_after_generate');

            if (seedWidget) toggleWidget(node, seedWidget, showSeedControls);
            if (controlWidget) toggleWidget(node, controlWidget, showSeedControls);
        }
    }

    updateNodeHeight(node);
}

app.registerExtension({
    name: "ComfyUI.Jimeng.DynamicWidgets",

    async setup() {
        console.log("%c[Jimeng] Dynamic Widgets Extension Loaded", "color:green; font-weight:bold;");
    },

    nodeCreated(node) {
        if (!node.comfyClass.startsWith("Jimeng")) return;

        const widgetsToWatch = node.widgets?.filter(w => TARGET_WIDGETS.includes(w.name));

        if (!widgetsToWatch || widgetsToWatch.length === 0) return;

        widgetsToWatch.forEach(w => {
            // 延迟初始化以确保组件就绪
            setTimeout(() => {
                widgetLogic(node, w);
            }, 100);

            // 监听值变更
            let widgetValue = w.value;
            Object.defineProperty(w, 'value', {
                get() {
                    return widgetValue;
                },
                set(newVal) {
                    if (newVal !== widgetValue) {
                        widgetValue = newVal;
                        widgetLogic(node, w);
                    }
                }
            });
        });
    }
});