from moviepy.editor import VideoFileClip, clips_array, vfx, TextClip, CompositeVideoClip

def v_concat():
    width = 864*2
    height = 486*2
    clip1 = VideoFileClip("video/480p.mp4").resize(2)
    clip2 = VideoFileClip("output/output_video.mp4")
    clip3 = VideoFileClip("video/720p.mp4").resize(1.35)
    label1 = "480p"
    label2 = "SR"
    label3 = "720p"
    # 添加标签
    txt_clip1 = TextClip(label1, fontsize=46, color='white')
    txt_clip2 = TextClip(label2, fontsize=46, color='white')
    txt_clip3 = TextClip(label3, fontsize=46, color='white')
                    
        
    
    vid_clip = clips_array([[clip1], [clip2]])
    final_clip = CompositeVideoClip([vid_clip,
                                     txt_clip1.set_position(('center', height-50)).set_duration(clip1.duration),
                                     txt_clip2.set_position(('center', height*2-50)).set_duration(clip2.duration)])

    final_clip.resize(width=864*2).write_videofile("output/compared.mp4")

if __name__ == '__main__':
    v_concat()