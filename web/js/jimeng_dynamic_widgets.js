import { app } from "/scripts/app.js";

// 定义需监听的组件名称列表
const TARGET_WIDGETS = [
    'size', 
    'enable_group_generation', 
    'generation_count', 
    'enable_timeout_setting', 
    'enable_random_seed', 
    'auto_duration',
    'draft_mode',
    'reuse_last_draft_task'
];

/**
 * 根据名称查找组件实例
 * @param {object} node - 节点实例
 * @param {string} name - 组件名称
 * @returns {object|null} - 找到的组件或 null
 */
function findWidgetByName(node, name) {
    if (!node.widgets) return null;
    return node.widgets.find((w) => w.name === name);
}

/**
 * 切换组件可见性状态
 * @param {object} node - 节点实例
 * @param {object} widget - 目标组件
 * @param {boolean} show - 是否显示
 * @returns {boolean} - 状态是否发生实质变更
 */
function toggleWidget(node, widget, show) {
    if (!widget) return false;

    // 缓存组件原始类型与尺寸计算方法
    if (!widget.origType && widget.type !== "hidden") {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
    }

    // 若无原始状态缓存且当前已隐藏，无法恢复，视为无变更
    if (!widget.origType && widget.type === "hidden") {
        return false;
    }

    // 检查目标状态与当前状态是否一致，若一致则无需变更
    const isCurrentlyHidden = widget.type === "hidden";
    if (show !== isCurrentlyHidden) return false;

    // 执行状态切换
    if (show) {
        widget.type = widget.origType;
        widget.computeSize = widget.origComputeSize;
    } else {
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    }

    return true;
}

/**
 * 更新节点高度，并保留用户手动调整的额外空间
 * @param {object} node - 节点实例
 * @param {number} extraHeight - 用户手动拉伸的额外高度补偿值
 */
function updateNodeHeight(node, extraHeight = 0) {
    if (node.flags?.collapsed) return;

    // 计算基础最小所需尺寸
    const size = node.computeSize();
    
    // 叠加用户手动调整的高度差，确保布局变更时不丢失拉伸空间
    const targetHeight = size[1] + extraHeight;

    node.setSize([node.size[0], targetHeight]);
    app.graph.setDirtyCanvas(true, true);
}

/**
 * 执行组件联动逻辑
 * @param {object} node - 节点实例
 * @param {object} widget - 触发变更的组件
 */
function widgetLogic(node, widget) {
    // 1. 计算布局变更前的高度差（Current Height Delta）
    // 用于在重绘时恢复用户手动拉伸的尺寸
    let extraHeight = 0;
    if (node.size && node.computeSize && !node.flags?.collapsed) {
        const currentMinHeight = node.computeSize()[1];
        const currentActualHeight = node.size[1];
        // 修正非正值，防止计算异常
        extraHeight = Math.max(0, currentActualHeight - currentMinHeight);
    }

    let shouldResize = false; 

    // 处理图像生成节点逻辑
    if (node.comfyClass === "JimengSeedream3" || node.comfyClass === "JimengSeedream4") {
        if (widget.name === 'size') {
            const isCustom = widget.value === "Custom";
            const widthWidget = findWidgetByName(node, 'width');
            const heightWidget = findWidgetByName(node, 'height');

            const changedW = toggleWidget(node, widthWidget, isCustom);
            const changedH = toggleWidget(node, heightWidget, isCustom);
            
            if (changedW || changedH) shouldResize = true;
        }
    }

    // 处理分组生成逻辑
    if (node.comfyClass === "JimengSeedream4") {
        if (widget.name === 'enable_group_generation') {
            const isGroupMode = widget.value === true;
            const maxImagesWidget = findWidgetByName(node, 'max_images');

            if (toggleWidget(node, maxImagesWidget, isGroupMode)) shouldResize = true;
        }
    }

    // 处理视频生成节点逻辑
    if (node.comfyClass === "JimengSeedance1" || 
        node.comfyClass === "JimengReferenceImage2Video" || 
        node.comfyClass === "JimengSeedance1_5") {

        // 1.5版本特定逻辑：智能时长控制 & 样片模式联动
        if (node.comfyClass === "JimengSeedance1_5") {
            if (widget.name === 'auto_duration') {
                const isAuto = widget.value === true;
                const durationWidget = findWidgetByName(node, 'duration');
                if (toggleWidget(node, durationWidget, !isAuto)) shouldResize = true;
            }

            if (widget.name === 'draft_mode') {
                const isDraftMode = widget.value === true;
                const draftTaskWidget = findWidgetByName(node, 'draft_task_id');
                const reuseWidget = findWidgetByName(node, 'reuse_last_draft_task');
                
                
                if (toggleWidget(node, reuseWidget, isDraftMode)) shouldResize = true;
                
                if (isDraftMode) {
                    if (reuseWidget) {
                         widgetLogic(node, reuseWidget);
                    } else {
                        if (toggleWidget(node, draftTaskWidget, true)) shouldResize = true;
                    }
                } else {
                    if (toggleWidget(node, draftTaskWidget, false)) shouldResize = true;
                }
            }

            if (widget.name === 'reuse_last_draft_task') {
                const isReuse = widget.value === true;
                const draftTaskWidget = findWidgetByName(node, 'draft_task_id');
                const draftModeWidget = findWidgetByName(node, 'draft_mode');
                const isDraftMode = draftModeWidget ? draftModeWidget.value === true : false;
                
                if (isDraftMode) {
                    if (toggleWidget(node, draftTaskWidget, !isReuse)) shouldResize = true;
                } else {
                     if (toggleWidget(node, draftTaskWidget, false)) shouldResize = true;
                }
            }
        }

        // 通用逻辑：批量生成选项联动
        if (widget.name === 'generation_count') {
            const isBatch = widget.value > 1;
            const batchPathWidget = findWidgetByName(node, 'filename_prefix');
            const saveLastFrameWidget = findWidgetByName(node, 'save_last_frame_batch');

            const changedPath = toggleWidget(node, batchPathWidget, isBatch);
            const changedSave = toggleWidget(node, saveLastFrameWidget, isBatch);
            
            if (changedPath || changedSave) shouldResize = true;
        }

        // 通用逻辑：随机种子控制联动
        if (widget.name === 'enable_random_seed') {
            const useRandom = widget.value === true;
            const showSeedControls = !useRandom;
            
            const seedWidget = findWidgetByName(node, 'seed');
            const controlWidget = findWidgetByName(node, 'control_after_generate');
            
            const changedSeed = toggleWidget(node, seedWidget, showSeedControls);
            const changedControl = toggleWidget(node, controlWidget, showSeedControls);
            
            if (changedSeed || changedControl) shouldResize = true;
        }
    }

    // 2. 若检测到布局实质变更，应用新尺寸并恢复高度差
    if (shouldResize) {
        updateNodeHeight(node, extraHeight);
    }
}

app.registerExtension({
    name: "ComfyUI.Jimeng.DynamicWidgets",

    async setup() {
        console.log("%c[Jimeng] Dynamic Widgets Extension Loaded", "color:green; font-weight:bold;");
    },

    nodeCreated(node) {
        if (!node.comfyClass.startsWith("Jimeng")) return;

        // 筛选需监听的组件
        const widgetsToWatch = node.widgets?.filter(w => TARGET_WIDGETS.includes(w.name));

        if (!widgetsToWatch || widgetsToWatch.length === 0) return;

        widgetsToWatch.forEach(w => {
            // 延迟初始化以确保组件状态就绪
            setTimeout(() => {
                widgetLogic(node, w);
            }, 100);

            // 劫持 value 属性以监听变更
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