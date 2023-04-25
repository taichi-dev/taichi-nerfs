//
//  ViewController.m
//  TaichiNerfTestbench
//
//  Created by PENGUINLIONG on 2023/2/16.
//

#import "ViewController.h"
#include "app_fp32.hpp"
#include "unistd.h"

std::vector<unsigned char> imagePostProcessing(const std::vector<float>& img,
                                               int width,
                                               int height) {
    std::vector<unsigned char> rgba(width * height * 4);

    for(int i=0; i < width * height; ++i) {
          rgba[4*i] = uint8_t(img[3*i] * 255);
          rgba[4*i+1] = uint8_t(img[3*i+1] * 255);
          rgba[4*i+2] = uint8_t(img[3*i+2] * 255);
          rgba[4*i+3] = 255;
    }
      
    return rgba;
}

UIImage* imageShow(unsigned char *rgba,
                      CGFloat width,
                      CGFloat height) {
    
    int bytes_per_pix = 4;
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

    CGContextRef newContext = CGBitmapContextCreate(rgba,
                                                    width, height, 8,
                                                    width * bytes_per_pix,
                                                    colorSpace, kCGImageAlphaNoneSkipLast);

    CGImageRef frame = CGBitmapContextCreateImage(newContext);
    
    UIImage *image = [UIImage imageWithCGImage:frame];
    
    CGImageRelease(frame);

    CGContextRelease(newContext);

    CGColorSpaceRelease(colorSpace);
    
    return image;
}

static App_nerf_f32 app_f32 = App_nerf_f32(TI_ARCH_METAL);

static double touchBeginLocationX = 0.0;
static double touchBeginLocationY = 0.0;

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    // Initialize AOT Module
    NSString *bundleRoot =[[NSBundle mainBundle] resourcePath];
    NSString *aotFilePath = [bundleRoot stringByAppendingPathComponent:@"taichi_ngp/compiled"];
    NSString *hashEmbeddingFilePath = [aotFilePath stringByAppendingPathComponent:@"hash_embedding.bin"];
    NSString *sigmaWeightsFilePath = [aotFilePath stringByAppendingPathComponent:@"sigma_weights.bin"];
    NSString *rgbWeightsFilePath = [aotFilePath stringByAppendingPathComponent:@"rgb_weights.bin"];
    NSString *densityBitfieldFilePath = [aotFilePath stringByAppendingPathComponent:@"density_bitfield.bin"];
    NSString *poseFilePath = [aotFilePath stringByAppendingPathComponent:@"pose.bin"];
    NSString *directionsFilePath = [aotFilePath stringByAppendingPathComponent:@"directions.bin"];
    
    // Modify Width & Height to stay consistent with what used in the taichi code
    // In this demo, we used 300 x 600 since it's generated from:
    //      python3 taichi_ngp.py --scene smh_lego --aot --res_w=300 --res_h=600
    //
    // If you would like to use an alternative resolution,
    // regenerate AOT files with correspondant "--res_w=... --res_h=..."
    //
    // iPad M1 Pro Max: 683 x 512
    // iPhone 14: 600 x 300
    int img_width = 300;
    int img_height = 600;
    app_f32.initialize(img_width, img_height,
                       std::string([aotFilePath UTF8String]),
                       std::string([hashEmbeddingFilePath UTF8String]),
                       std::string([sigmaWeightsFilePath UTF8String]),
                       std::string([rgbWeightsFilePath UTF8String]),
                       std::string([densityBitfieldFilePath UTF8String]),
                       std::string([poseFilePath UTF8String]),
                       std::string([directionsFilePath UTF8String]));

    app_f32.pose_rotate_scale(0.0, 0.0, 0.0, 2.5);
    std::vector<float> img = app_f32.run();
    auto img_data = imagePostProcessing(img, app_f32.kWidth, app_f32.kHeight);
    UIImage* image = imageShow(img_data.data(), app_f32.kWidth, app_f32.kHeight);
    UIImageView* imageView = [[UIImageView alloc] initWithImage:image];
    
    [imageView setFrame:CGRectMake(0, 0, image.size.width, image.size.height)];
    imageView.frame = self.view.bounds;
    [self.view addSubview:imageView];
}

- (void) touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [super touchesBegan:touches withEvent:event];

    UITouch *touch = [[event allTouches] anyObject];
    CGPoint touchLocation = [touch locationInView:touch.view];
    
    touchBeginLocationX = touchLocation.x;
    touchBeginLocationY = touchLocation.y;
}

- (void) touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [super touchesEnded:touches withEvent:event];
    
    touchBeginLocationX = 0.0;
    touchBeginLocationY = 0.0;
}

- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [super touchesMoved:touches withEvent:event];
    
    int num_touches = [[event allTouches] count];
    
    float radius = 2.5;
    float angle_x = 0.0;
    float angle_y = 0.0;
    float angle_z = 0.0;
    const float base_angle = 0.1;
    
    // Setup "angle" and "radius" based on Touch Event
    if(num_touches == 1) {
        UITouch *touch = [[event allTouches] anyObject];
        CGPoint touchLocation = [touch locationInView:touch.view];
        
        double move_x = touchLocation.x - touchBeginLocationX;
        double move_y = touchLocation.y - touchBeginLocationY;
        
        if(std::abs(move_x) > std::abs(move_y)) {
            move_y = 0.0;
        } else {
            move_x = 0.0;
        }
        
        angle_x = base_angle * move_y * 0.01;
        angle_y = base_angle * move_x * 0.01;
        
    } else if(num_touches == 2) {
        UITouch *touch = [[event allTouches] anyObject];
        CGPoint touchLocation = [touch locationInView:touch.view];
        
        double move_x = touchLocation.x - touchBeginLocationX;
        angle_z = base_angle * move_x * 0.01;
    }
    
    // Rotate camera based on "angle" & "radius" collected from Touch Event
    app_f32.pose_rotate_scale(angle_x, angle_y, angle_z, radius);
    
    // Run Nerf Inference
    std::vector<float> img = app_f32.run();
    
    // Display output image
    auto img_data = imagePostProcessing(img, app_f32.kWidth, app_f32.kHeight);
    UIImage* image = imageShow(img_data.data(), app_f32.kWidth, app_f32.kHeight);
    UIImageView* imageView = [[UIImageView alloc] initWithImage:image];
    
    [imageView setFrame:CGRectMake(0, 0, image.size.width, image.size.height)];
    imageView.frame = self.view.bounds;
    [self.view addSubview:imageView];
    
}

@end
