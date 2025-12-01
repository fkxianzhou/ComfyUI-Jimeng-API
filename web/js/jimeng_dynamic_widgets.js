import { app } from "/scripts/app.js";

function findWidgetByName(node, name) {
    if (!node.widgets) return null;
    return node.widgets.find((w) => w.name === name);
}

function toggleWidget(node, widget, show) {
    if (!widget) return;

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

function updateNodeHeight(node) {
    if(node.flags && node.flags.collapsed) return;
    
    const size = node.computeSize();
    node.setSize([node.size[0], size[1]]);
    app.graph.setDirtyCanvas(true, true);
}

const TARGET_WIDGETS = ['size', 'generation_mode', 'generation_count'];

function widgetLogic(node, widget) {
    if (node.comfyClass === "JimengSeedream3" || node.comfyClass === "JimengSeedream4") {
        if (widget.name === 'size') {
            const isCustom = widget.value === "Custom";
            const widthWidget = findWidgetByName(node, 'width');
            const heightWidget = findWidgetByName(node, 'height');

            if (widthWidget) toggleWidget(node, widthWidget, isCustom);
            if (heightWidget) toggleWidget(node, heightWidget, isCustom);
        }
    }

    if (node.comfyClass === "JimengSeedream4") {
        if (widget.name === 'generation_mode') {
            const isGroupMode = widget.value === "Image Group (auto)";
            const maxImagesWidget = findWidgetByName(node, 'max_images');
            
            if (maxImagesWidget) toggleWidget(node, maxImagesWidget, isGroupMode);
        }
    }

    if (node.comfyClass === "JimengVideoGeneration" || node.comfyClass === "JimengReferenceImage2Video") {
        if (widget.name === 'generation_count') {
            const isBatch = widget.value > 1;
            
            const batchPathWidget = findWidgetByName(node, 'batch_save_path');
            const saveLastFrameWidget = findWidgetByName(node, 'save_last_frame_batch');

            if (batchPathWidget) toggleWidget(node, batchPathWidget, isBatch);
            if (saveLastFrameWidget) toggleWidget(node, saveLastFrameWidget, isBatch);
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
            setTimeout(() => {
                widgetLogic(node, w);
            }, 100);

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