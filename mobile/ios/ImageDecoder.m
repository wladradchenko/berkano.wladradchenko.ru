#import <React/RCTBridgeModule.h>
#import <UIKit/UIKit.h>

@interface RCT_EXTERN_MODULE(ImageDecoder, NSObject)
RCT_EXTERN_METHOD(decodeToTensor:(NSString *)uriString targetWidth:(NSInteger)targetWidth targetHeight:(NSInteger)targetHeight resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
@end
