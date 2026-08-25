import { app } from "/scripts/app.js";

function isVueNodesEnabled() {
    const settings = app?.ui?.settings;
    const getter = settings?.getSettingValue;
    if (typeof getter !== "function") return false;
    try {
        return getter.call(settings, "Comfy.VueNodes.Enabled", false) === true;
    } catch {
        return false;
    }
}

function getComfyLocale() {
    const settings = app?.ui?.settings;
    const getter = settings?.getSettingValue;
    if (typeof getter === "function") {
        try {
            const val = getter.call(settings, "Comfy.Locale", "");
            if (typeof val === "string" && val) return val;
        } catch {
        }
    }
    return (navigator.language || "en").split("-")[0];
}

function isZhLocale() {
    const locale = getComfyLocale();
    return locale === "zh" || locale.startsWith("zh-");
}

const VUE_NODES_ENABLED = isVueNodesEnabled();

/**
 * 定义需监听的组件名称列表
 * @type {string[]}
 */
const TARGET_WIDGETS = [
    'model_version',
    'size',
    'enable_group_generation',
    'generation_count',
    'enable_timeout_setting',
    'enable_random_seed',
    'auto_duration',
    'draft_mode',
    'reuse_last_draft_task',
    'key_name',
    'reasoning_mode',
    'thinking'
];

const LEGACY_WIDGET_ORDERS = {
    JimengSeedance2: [
        'model_version', 'prompt', 'enable_random_seed', 'seed',
        'control_after_generate', 'resolution', 'aspect_ratio', 'auto_duration',
        'duration', 'generate_audio', 'enable_web_search', 'generation_count',
        'filename_prefix', 'save_last_frame_batch', 'non_blocking'
    ],
    JimengSeedream5: [
        'model_version', 'prompt', 'size', 'width', 'height', 'seed',
        'control_after_generate', 'enable_group_generation', 'max_images',
        'enable_web_search', 'generation_count', 'watermark'
    ],
};

/**
 * 节点底部预留的缓冲高度
 * @type {number}
 */
const BOTTOM_PADDING = 8;

/**
 * 根据名称查找组件实例
 * @param {object} node - 节点实例
 * @param {string} name - 组件名称
 * @returns {object|null} 找到的组件或 null
 */
function findWidgetByName(node, name) {
    if (!node.widgets) return null;
    return node.widgets.find((w) =>
        w.name === name || String(w.name || '').endsWith(`.${name}`)
    );
}

function getWidgetBaseName(widget) {
    return String(widget?.name || '').split('.').pop();
}

/**
 * 检查节点是否存在同名输入且已被链接
 * @param {object} node - 节点实例
 * @param {string} name - 组件名称
 * @returns {boolean} 是否存在已链接的同名输入
 */
function isWidgetLinked(node, name) {
    return node.inputs ? node.inputs.some((input) => input.name === name && input.link != null) : false;
}

/**
 * 切换组件可见性状态
 * @param {object} node - 节点实例
 * @param {object} widget - 目标组件
 * @param {boolean} show - 是否显示
 * @returns {boolean} 状态是否发生实质变更
 */
function toggleWidget(node, widget, show) {
    if (!widget) return false;

    if (VUE_NODES_ENABLED) {
        const disabled = !show;
        const changed = widget.disabled !== disabled || widget.options?.disabled !== disabled;
        widget.disabled = disabled;
        widget.options = widget.options || {};
        widget.options.disabled = disabled;
        if (widget.inputEl) widget.inputEl.disabled = disabled;
        if (changed) app.graph.setDirtyCanvas(true, true);
        return changed;
    }

    // 如果组件已被转换为输入且已链接，则不进行隐藏操作
    if (isWidgetLinked(node, widget.name)) return false;

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
 * @param {number} [extraHeight=0] - 用户手动拉伸的额外高度补偿值
 */
function updateNodeHeight(node, extraHeight = 0) {
    if (VUE_NODES_ENABLED) return;
    if (node.flags?.collapsed) return;

    // 计算基础最小所需尺寸
    const size = node.computeSize();

    // 叠加用户手动调整的高度差，并加上底部缓冲
    const targetHeight = size[1] + extraHeight;

    node.setSize([node.size[0], targetHeight]);
    app.graph.setDirtyCanvas(true, true);
}

/**
 * 为最后一个可见组件注入底部缓冲
 * @param {object} node - 节点实例
 * @returns {boolean} 是否发生了变更
 */
function applyBottomPadding(node) {
    if (VUE_NODES_ENABLED) return false;
    if (!node.widgets) return false;

    // 1. 找到最后一个可见组件
    let lastWidget = null;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        if (node.widgets[i].type !== "hidden") {
            lastWidget = node.widgets[i];
            break;
        }
    }

    let changed = false;

    // 2. 清理之前注入的 Padding（如果有）
    node.widgets.forEach(w => {
        if (w !== lastWidget && w.hasBottomPadding) {
            w.computeSize = w.origComputeSizeBeforePadding;
            delete w.origComputeSizeBeforePadding;
            delete w.hasBottomPadding;
            changed = true;
        }
    });

    if (!lastWidget) return changed;

    // 3. 给当前的 lastWidget 注入 Padding
    if (!lastWidget.hasBottomPadding) {
        if (!lastWidget.computeSize) {
            // 如果没有 computeSize，创建一个默认的 guess
            lastWidget.computeSize = () => [0, 20];
        }

        lastWidget.origComputeSizeBeforePadding = lastWidget.computeSize;
        const originalMethod = lastWidget.computeSize;

        lastWidget.computeSize = function (...args) {
            const size = originalMethod.apply(this, args);
            // 确保返回新数组，并在高度上增加 Padding
            return [size ? size[0] : 0, (size ? size[1] : 20) + BOTTOM_PADDING];
        };

        lastWidget.hasBottomPadding = true;
        changed = true;
    }

    return changed;
}

const AUTOGROW_LABEL_RULES = {
    JimengSeedream4: [
        { prefix: "image_", zhLabel: "图像", enLabel: "Image" },
    ],
    JimengSeedream5: [
        { prefix: "image_", zhLabel: "图像", enLabel: "Image" },
    ],
    JimengSeedance2: [
        { prefix: "ref_image_", zhLabel: "参考图片", enLabel: "Ref Image" },
        { prefix: "ref_video_", zhLabel: "参考视频", enLabel: "Ref Video" },
        { prefix: "ref_audio_", zhLabel: "参考音频", enLabel: "Ref Audio" },
    ],
};

// ComfyUI frontend 1.45.x creates DynamicCombo children with names such as
// `model_version.seed`. The normal locale keys are therefore not reused for
// those children. Keep a small runtime fallback for Vue Nodes and for older
// frontends that do not refresh translated labels after a model switch.
const DYNAMIC_WIDGET_ZH_LABELS = {
    JimengSeedream5: {
        prompt: "提示词",
        size: "尺寸",
        width: "宽度",
        height: "高度",
        seed: "种子",
        enable_group_generation: "启用组图生成",
        max_images: "最大图像数",
        enable_web_search: "启用联网搜索",
        thinking: "提示词优化",
        generation_count: "生成数量",
        watermark: "水印",
    },
    JimengSeedance2: {
        prompt: "提示词",
        enable_random_seed: "使用随机种子",
        seed: "种子",
        resolution: "分辨率",
        aspect_ratio: "宽高比",
        auto_duration: "智能时长",
        duration: "时长 (秒)",
        generate_audio: "生成音频",
        enable_web_search: "联网搜索",
        generation_count: "生成数量",
        filename_prefix: "文件名前缀",
        save_last_frame_batch: "保存尾帧",
        non_blocking: "非阻塞模式",
    },
};

const DYNAMIC_WIDGET_ZH_TOOLTIPS = {
    JimengSeedream5: {
        thinking: "开启后会在生成前优化提示词；Seedream 5 Pro 使用参考图时必须开启。",
    },
    JimengSeedance2: {
        resolution: "Seedance 2.5 支持 480p、720p；Seedance 2.0 标准版最高支持 4K。",
        duration: "Seedance 2.5 支持 4–30 秒；Seedance 2.0 系列支持 4–15 秒。",
    },
};

function getAutogrowLabelRules(node) {
    if (!node || !node.comfyClass) return [];
    return AUTOGROW_LABEL_RULES[node.comfyClass] || [];
}

function escapeRegex(text) {
    return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function setInputDisplayText(input, text) {
    const changed =
        input.label !== text ||
        input.localized_name !== text ||
        input.display_name !== text;
    input.label = text;
    input.localized_name = text;
    input.display_name = text;
    return changed;
}

function applyAutogrowInputLabels(node) {
    if (!node.inputs || node.inputs.length === 0) return false;
    const rules = getAutogrowLabelRules(node);
    if (!rules || rules.length === 0) return false;

    const isZh = isZhLocale();
    let changed = false;

    for (const input of node.inputs) {
        if (!input || !input.name) continue;
        const inputName = String(input.name);
        for (const rule of rules) {
            // 兼容 Autogrow 生成的命名形式
            const pattern = new RegExp(`(?:^|\\.)${escapeRegex(rule.prefix)}(\\d+)$`);
            const matched = inputName.match(pattern);
            if (!matched) continue;
            const suffixNum = parseInt(matched[1], 10);
            if (!Number.isFinite(suffixNum)) continue;
            const displayLabel = isZh ? rule.zhLabel : rule.enLabel;
            const expectedLabel = `${displayLabel} ${suffixNum}`;
            if (setInputDisplayText(input, expectedLabel)) {
                changed = true;
            }
            break;
        }
    }
    return changed;
}

function refreshAutogrowInputLabels(node) {
    if (applyAutogrowInputLabels(node)) {
        app.graph.setDirtyCanvas(true, true);
    }
}

function restoreLegacyWidgetValues(node, data) {
    const order = LEGACY_WIDGET_ORDERS[node.comfyClass];
    const values = data?.widgets_values;
    if (!order || !Array.isArray(values) || values.length < 2) return;
    if (node.comfyClass === 'JimengSeedream5' && values[0] === 'doubao-seedream-5.0-pro') {
        setTimeout(() => installWidgetWatchers(node), 0);
        setTimeout(() => {
            installWidgetWatchers(node);
            if (Array.isArray(data?.size) && data.size.length >= 2) {
                node.setSize?.([data.size[0], data.size[1]]);
            }
        }, 120);
        return;
    }

    const legacyValues = new Map();
    order.forEach((name, index) => {
        if (index < values.length) legacyValues.set(name, values[index]);
    });

    const applyValues = () => {
        for (const [name, value] of legacyValues.entries()) {
            const target = findWidgetByName(node, name);
            if (target && value !== undefined && target.value !== value) {
                target.value = value;
            }
        }
    };

    const migrateLinkedInputs = () => {
        if (!node.inputs) return;
        for (const name of order.slice(1)) {
            const legacyIndex = node.inputs.findIndex((input) => input.name === name);
            const dynamicIndex = node.inputs.findIndex(
                (input) => input.name === `model_version.${name}`
            );
            if (legacyIndex < 0 || dynamicIndex < 0 || legacyIndex === dynamicIndex) continue;

            const legacyInput = node.inputs[legacyIndex];
            const dynamicInput = node.inputs[dynamicIndex];
            if (legacyInput.link != null && dynamicInput.link == null) {
                dynamicInput.link = legacyInput.link;
            }
            node.inputs.splice(legacyIndex, 1);
        }

        node.inputs.forEach((input, index) => {
            if (input.link == null) return;
            const links = node.graph?.links;
            const link = links?.get?.(input.link) ?? links?.[input.link];
            if (link) link.target_slot = index;
        });
        node._setConcreteSlots?.();
    };

    const restoreSize = () => {
        if (!Array.isArray(data?.size) || data.size.length < 2) return;
        node.setSize?.([data.size[0], data.size[1]]);
    };

    const modelWidget = findWidgetByName(node, 'model_version');
    if (modelWidget && legacyValues.has('model_version')) {
        const selectedModel = legacyValues.get('model_version');
        modelWidget.value = selectedModel;
    }
    queueMicrotask(applyValues);
    setTimeout(applyValues, 0);
    setTimeout(() => {
        applyValues();
        migrateLinkedInputs();
        restoreSize();
    }, 120);
    setTimeout(() => installWidgetWatchers(node), 0);
    setTimeout(() => installWidgetWatchers(node), 120);
}

function refreshReferenceThinkingState(node) {
    if (node.comfyClass !== 'JimengSeedream5') return;
    const thinkingWidget = findWidgetByName(node, 'thinking');
    if (!thinkingWidget) return;
    const hasReference = (node.inputs || []).some((input) =>
        /(?:^|\.)image_\d+$/.test(String(input?.name || '')) && input.link != null
    );
    if (hasReference && thinkingWidget.value !== true) thinkingWidget.value = true;
    toggleWidget(node, thinkingWidget, !hasReference);
}

/**
 * 执行组件联动逻辑
 * @param {object} node - 节点实例
 * @param {object} widget - 触发变更的组件
 */
function widgetLogic(node, widget) {
    const widgetName = getWidgetBaseName(widget);
    // 1. 计算布局变更前的高度差（Current Height Delta）
    // 用于在重绘时恢复用户手动拉伸的尺寸
    let extraHeight = 0;
    if (!VUE_NODES_ENABLED && node.size && node.computeSize && !node.flags?.collapsed) {
        const currentMinHeight = node.computeSize()[1];
        const currentActualHeight = node.size[1];
        // 修正非正值，防止计算异常
        extraHeight = Math.max(0, currentActualHeight - currentMinHeight);
    }

    let shouldResize = false;

    // 处理图像生成节点逻辑
    if (node.comfyClass === "JimengSeedream3" || node.comfyClass === "JimengSeedream4" || node.comfyClass === "JimengSeedream5") {
        if (widgetName === 'size') {
            const isCustom = widget.value === "Custom";
            const widthWidget = findWidgetByName(node, 'width');
            const heightWidget = findWidgetByName(node, 'height');

            const changedW = toggleWidget(node, widthWidget, isCustom);
            const changedH = toggleWidget(node, heightWidget, isCustom);

            if (changedW || changedH) shouldResize = true;
        }

        if (node.comfyClass === "JimengSeedream4" && widgetName === 'model_version') {
            const optimizePromptWidget = findWidgetByName(node, 'thinking');
            const isSupported = widget.value === "doubao-seedream-4.0";
            if (toggleWidget(node, optimizePromptWidget, isSupported)) shouldResize = true;
        }
    }

    // 处理分组生成逻辑
    if (node.comfyClass === "JimengSeedream4" || node.comfyClass === "JimengSeedream5") {
        if (widgetName === 'enable_group_generation') {
            const isGroupMode = widget.value === true;
            const maxImagesWidget = findWidgetByName(node, 'max_images');

            if (toggleWidget(node, maxImagesWidget, isGroupMode)) shouldResize = true;
        }
    }

    // 处理视频生成节点逻辑
    if (node.comfyClass === "JimengSeedance1" ||
        node.comfyClass === "JimengReferenceImage2Video" ||
        node.comfyClass === "JimengSeedance1_5" ||
        node.comfyClass === "JimengSeedance2") {

        // 1.5/2.0 版本特定逻辑：智能时长控制
        if (node.comfyClass === "JimengSeedance1_5" || node.comfyClass === "JimengSeedance2") {
            if (widgetName === 'auto_duration') {
                const isAuto = widget.value === true;
                const durationWidget = findWidgetByName(node, 'duration');
                if (toggleWidget(node, durationWidget, !isAuto)) shouldResize = true;
            }
        }

        // 1.5版本特定逻辑：样片模式联动
        if (node.comfyClass === "JimengSeedance1_5") {
            if (widgetName === 'draft_mode') {
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

            if (widgetName === 'reuse_last_draft_task') {
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
        if (widgetName === 'generation_count') {
            const isBatch = widget.value > 1;
            const batchPathWidget = findWidgetByName(node, 'filename_prefix');
            const saveLastFrameWidget = findWidgetByName(node, 'save_last_frame_batch');

            const changedPath = toggleWidget(node, batchPathWidget, isBatch);
            const changedSave = toggleWidget(node, saveLastFrameWidget, isBatch);

            if (changedPath || changedSave) shouldResize = true;
        }

        // 通用逻辑：随机种子控制联动
        if (widgetName === 'enable_random_seed') {
            const useRandom = widget.value === true;
            const showSeedControls = !useRandom;

            const seedWidget = findWidgetByName(node, 'seed');
            const controlWidget = findWidgetByName(node, 'control_after_generate');

            const changedSeed = toggleWidget(node, seedWidget, showSeedControls);
            const changedControl = toggleWidget(node, controlWidget, showSeedControls);

            if (changedSeed || changedControl) shouldResize = true;
        }
    }

    // 处理 API Client 节点逻辑
    if (node.comfyClass === "JimengAPIClient") {
        if (widgetName === 'key_name') {
            const isCustom = widget.value === "Custom";
            const newKeyWidget = findWidgetByName(node, 'new_api_key');
            const newNameWidget = findWidgetByName(node, 'new_key_name');

            const changedKey = toggleWidget(node, newKeyWidget, isCustom);
            const changedName = toggleWidget(node, newNameWidget, isCustom);

            if (changedKey || changedName) shouldResize = true;
        }
    }

    // 处理视觉理解节点逻辑
    if (node.comfyClass === "JimengVisualUnderstanding") {
        if (widgetName === 'reasoning_mode') {
            const isThinkingEnabled = widget.value !== "disabled";
            const effortWidget = findWidgetByName(node, 'reasoning_effort');
            
            if (toggleWidget(node, effortWidget, isThinkingEnabled)) shouldResize = true;
        }
    }

    refreshReferenceThinkingState(node);

    // 2. 若检测到布局实质变更，应用新尺寸并恢复高度差
    const paddingChanged = applyBottomPadding(node);

    // 如果正在配置中（如从 workflow 加载），则跳过高度调整，避免覆盖用户保存的尺寸
    if (node._isConfiguring) return;

    if (shouldResize || paddingChanged) {
        updateNodeHeight(node, extraHeight);
    }
}

function isUntranslatedDynamicLabel(value, fullName, baseName) {
    return value == null || value === "" || value === fullName || value === baseName;
}

function applyDynamicWidgetLabels(node) {
    if (!isZhLocale()) return false;
    const labels = DYNAMIC_WIDGET_ZH_LABELS[node?.comfyClass];
    if (!labels) return false;
    const tooltips = DYNAMIC_WIDGET_ZH_TOOLTIPS[node?.comfyClass] || {};

    let changed = false;
    for (const input of node.inputs || []) {
        const fullName = String(input?.name || "");
        if (!fullName.startsWith("model_version.")) continue;
        const baseName = fullName.slice("model_version.".length);
        const label = labels[baseName];
        if (!label) continue;

        for (const property of ["label", "localized_name", "display_name"]) {
            if (isUntranslatedDynamicLabel(input[property], fullName, baseName)) {
                input[property] = label;
                changed = true;
            }
        }
        if (tooltips[baseName] && input.tooltip !== tooltips[baseName]) {
            input.tooltip = tooltips[baseName];
            changed = true;
        }
    }

    for (const widget of node.widgets || []) {
        const fullName = String(widget?.name || "");
        if (!fullName.startsWith("model_version.")) continue;
        const baseName = fullName.slice("model_version.".length);
        const label = labels[baseName];
        if (!label) continue;

        for (const property of ["label", "localized_name", "display_name"]) {
            if (isUntranslatedDynamicLabel(widget[property], fullName, baseName)) {
                widget[property] = label;
                changed = true;
            }
        }
        if (tooltips[baseName] && widget.tooltip !== tooltips[baseName]) {
            widget.tooltip = tooltips[baseName];
            changed = true;
        }
    }

    return changed;
}

function refreshDynamicWidgetLabels(node) {
    if (applyDynamicWidgetLabels(node)) {
        app.graph.setDirtyCanvas(true, true);
    }
}

function migrateSeedream4WidgetValues(node, data) {
    if (node.comfyClass !== 'JimengSeedream4' || !Array.isArray(data?.widgets_values)) {
        return data;
    }

    const values = data.widgets_values;
    let migratedValues = null;

    // Workflows saved before the prompt-optimization control existed.
    if (values.length === 10 || values.length === 11) {
        const insertionIndex = values.length - 1;
        migratedValues = [
            ...values.slice(0, insertionIndex),
            false,
            ...values.slice(insertionIndex),
        ];
    }

    // Workflows saved while the Boolean control was temporarily placed after
    // size or seed. Move it directly above watermark.
    if (
        values.length >= 12 &&
        typeof values[10] !== 'boolean'
    ) {
        const sourceIndex = typeof values[3] === 'boolean'
            ? 3
            : (typeof values[7] === 'boolean' ? 7 : -1);
        if (sourceIndex >= 0) {
            const thinking = values[sourceIndex];
            const valuesWithoutThinking = [
                ...values.slice(0, sourceIndex),
                ...values.slice(sourceIndex + 1),
            ];
            const insertionIndex = valuesWithoutThinking.length - 1;
            migratedValues = [
                ...valuesWithoutThinking.slice(0, insertionIndex),
                thinking,
                ...valuesWithoutThinking.slice(insertionIndex),
            ];
        }
    }

    // Workflows briefly saved with the three-state mode control at the end.
    const legacyMode = values[values.length - 1];
    if (
        typeof legacyMode === 'string' &&
        ['disabled', 'standard', 'fast'].includes(legacyMode)
    ) {
        const valuesWithoutMode = values.slice(0, -1);
        const insertionIndex = valuesWithoutMode.length - 1;
        migratedValues = [
            ...valuesWithoutMode.slice(0, insertionIndex),
            legacyMode !== 'disabled',
            ...valuesWithoutMode.slice(insertionIndex),
        ];
    }

    return migratedValues ? { ...data, widgets_values: migratedValues } : data;
}

function migrateSeedream5WidgetValues(node, data) {
    const values = data?.widgets_values;
    if (
        node.comfyClass !== 'JimengSeedream5' ||
        !Array.isArray(values) ||
        values[0] !== 'doubao-seedream-5.0-pro'
    ) {
        return data;
    }

    // Previous Pro order: seed control -> thinking -> generation count -> watermark.
    if (typeof values[7] === 'boolean' && typeof values[8] === 'number') {
        const migratedValues = [
            ...values.slice(0, 7),
            values[8],
            values[7],
            ...values.slice(9),
        ];
        return { ...data, widgets_values: migratedValues };
    }

    return data;
}

function installWidgetWatchers(node) {
    refreshDynamicWidgetLabels(node);
    const widgetsToWatch = node.widgets?.filter(w =>
        TARGET_WIDGETS.includes(getWidgetBaseName(w))
    );
    if (!widgetsToWatch || widgetsToWatch.length === 0) return;

    widgetsToWatch.forEach(w => {
        if (w._jimengValueWatcherInstalled) return;
        w._jimengValueWatcherInstalled = true;
        widgetLogic(node, w);

        if (getWidgetBaseName(w) === 'model_version') {
            const originalCallback = w.callback;
            w.callback = function (...args) {
                const result = originalCallback?.apply(this, args);
                widgetLogic(node, w);
                queueMicrotask(() => installWidgetWatchers(node));
                setTimeout(() => installWidgetWatchers(node), 0);
                setTimeout(() => installWidgetWatchers(node), 120);
                return result;
            };
            return;
        }

        let widgetValue = w.value;
        try {
            Object.defineProperty(w, 'value', {
                configurable: true,
                enumerable: true,
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
        } catch {
            delete w._jimengValueWatcherInstalled;
        }
    });
}

app.registerExtension({
    name: "ComfyUI.Jimeng.DynamicWidgets",

    async setup() {
        const mode = VUE_NODES_ENABLED ? "Node2.0(Vue)" : "Legacy(Canvas)";
        console.log(`%c[Jimeng] Dynamic Widgets Extension Loaded (${mode})`, "color:green; font-weight:bold;");
    },

    nodeCreated(node) {
        if (!node.comfyClass || !node.comfyClass.startsWith("Jimeng")) return;

        // 劫持 configure 方法以检测是否处于加载/配置阶段
        const origConfigure = node.configure;
        node.configure = function (data) {
            this._isConfiguring = true;
            let migratedData = migrateSeedream4WidgetValues(this, data);
            migratedData = migrateSeedream5WidgetValues(this, migratedData);
            const configureArgs = [migratedData, ...Array.from(arguments).slice(1)];
            const r = origConfigure ? origConfigure.apply(this, configureArgs) : undefined;
            restoreLegacyWidgetValues(this, migratedData);
            refreshDynamicWidgetLabels(this);
            setTimeout(() => refreshDynamicWidgetLabels(this), 0);
            setTimeout(() => refreshDynamicWidgetLabels(this), 120);
            delete this._isConfiguring;
            return r;
        };

        const onConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link_info, slot) {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            if (!this._isConfiguring) {
                refreshAutogrowInputLabels(this);
                refreshReferenceThinkingState(this);
            }
            return r;
        };

        refreshAutogrowInputLabels(node);
        setTimeout(() => refreshAutogrowInputLabels(node), 0);
        setTimeout(() => refreshAutogrowInputLabels(node), 120);
        setTimeout(() => refreshAutogrowInputLabels(node), 360);

        refreshDynamicWidgetLabels(node);
        setTimeout(() => refreshDynamicWidgetLabels(node), 0);
        setTimeout(() => refreshDynamicWidgetLabels(node), 120);
        setTimeout(() => refreshDynamicWidgetLabels(node), 360);

        installWidgetWatchers(node);
        setTimeout(() => installWidgetWatchers(node), 0);
        setTimeout(() => installWidgetWatchers(node), 120);
    }
});
