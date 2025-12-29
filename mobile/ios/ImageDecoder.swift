import Foundation
import UIKit

@objc(ImageDecoder)
class ImageDecoder: NSObject {

  @objc
  func decodeToTensor(_ uriString: String, targetWidth: Int, targetHeight: Int, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {

    guard let url = URL(string: uriString),
          let data = try? Data(contentsOf: url),
          let image = UIImage(data: data) else {
      reject("ERROR", "Cannot load image", nil)
      return
    }

    // Resize
    UIGraphicsBeginImageContext(CGSize(width: targetWidth, height: targetHeight))
    image.draw(in: CGRect(x:0, y:0, width: targetWidth, height: targetHeight))
    let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()

    guard let cgImage = resizedImage?.cgImage else {
      reject("ERROR", "Cannot get CGImage", nil)
      return
    }

    let width = cgImage.width
    let height = cgImage.height
    var tensor = [Float](repeating: 0, count: width*height*3)

    let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width*4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    )!

    context.draw(cgImage, in: CGRect(x:0, y:0, width: width, height: height))
    guard let pixelData = context.data else {
      reject("ERROR", "Cannot get pixel data", nil)
      return
    }

    let ptr = pixelData.bindMemory(to: UInt8.self, capacity: width*height*4)
    for y in 0..<height {
      for x in 0..<width {
        let idx = y*width + x
        let r = ptr[idx*4]
        let g = ptr[idx*4+1]
        let b = ptr[idx*4+2]
        tensor[idx] = Float(r)/255.0
        tensor[width*height + idx] = Float(g)/255.0
        tensor[2*width*height + idx] = Float(b)/255.0
      }
    }

    resolve(tensor)
  }
}
