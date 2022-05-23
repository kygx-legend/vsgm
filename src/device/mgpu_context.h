#pragma once

#include <moderngpu/launch_box.hxx>

// launch box setting
// 3090
// typedef mgpu::launch_box_t<mgpu::arch_86_cta<256, 8, 8>, mgpu::arch_86_cta<256, 8, 8>> MGPULaunchBox;
// typedef mgpu::launch_box_t<mgpu::arch_86_cta<256, 1>, mgpu::arch_86_cta<256, 1>> MGPULaunchBoxVT1;

// 2080 ti and t4
typedef mgpu::launch_box_t<mgpu::arch_75_cta<128, 8, 8>, mgpu::arch_75_cta<128, 8, 8>> MGPULaunchBox;
typedef mgpu::launch_box_t<mgpu::arch_75_cta<128, 1>, mgpu::arch_75_cta<128, 1>> MGPULaunchBoxVT1;

// V100
// typedef mgpu::launch_box_t<mgpu::arch_70_cta<128, 8, 8>, mgpu::arch_70_cta<128, 8, 8>> MGPULaunchBox;
// typedef mgpu::launch_box_t<mgpu::arch_70_cta<128, 1>, mgpu::arch_70_cta<128, 1>> MGPULaunchBoxVT1;
