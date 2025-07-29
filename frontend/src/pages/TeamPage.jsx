import React from 'react';
import { Mail, Linkedin, Github, Users } from 'lucide-react';

const TeamPage = () => {
  const teamMembers = [
    {
      name: "Bilal Khan",
      role: "AI/ML Expert",
      image: "https://placehold.co/128x128/FF5733/FFFFFF?text=BK",
      bio: "Learner under the guidance of abhisheksingh. Passionate about AI and its applications in insurance.",
      linkedin: "#",
      github: "#",
      email: "khanbilal080804@gmail.com"
    },
    {
      name: "Prabhat Singh",
      role: "API Developer & Backend Engineer",
      image: "https://placehold.co/128x128/33FF57/FFFFFF?text=PS",
      bio: "learner under the guidance of abhisheksingh. Focused on backend systems and API development.",
      linkedin: "https://www.linkedin.com/in/prabhat-singh-736643287",
      github: "https://github.com/Prabhatsingh001",
      email: "prabhatsingh1112004@gmail.com"
    },
    {
      name: "Himanshu Patel",
      role: "API Developer & Backend Engineer",
      image: "https://placehold.co/128x128/3357FF/FFFFFF?text=HP",
      bio: "learner under the guidance of abhisheksingh. Specializes in backend development and API integration.",
      linkedin: "#",
      github: "#",
      email: "himanshupatel2125@gmail.com"
    },
    {
      name: "Abhisheksingh",
      role: "Full Stack Technical Lead",
      image: "https://placehold.co/128x128/FF33A1/FFFFFF?text=ASD",
      bio: "Full Stack Developer and Technical Lead with a passion for AI and insurance technology. Leading the team to build innovative solutions.",
      linkedin: "www.linkedin.com/in/abhisheksingh-dhangar-8b19b6275",
      github: "https://github.com/dhangar007A",
      email: "abhisheksinghdhangar007@gmail.com"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-black pt-20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-16">
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent mb-6">
            Meet Our Team
          </h1>
          <p className="text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed">
            A diverse group of AI researchers, engineers, and insurance experts dedicated to revolutionizing policy management
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {teamMembers.map((member, index) => (
            <div key={index} className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-3xl p-8 border border-slate-700/30 hover:border-purple-500/30 transition-all duration-500 group">
              <div className="text-center mb-6">
                <div className="relative inline-block">
                  <img
                    src={member.image}
                    alt={member.name}
                    className="w-32 h-32 rounded-full mx-auto mb-4 border-4 border-gradient-to-r from-purple-400 to-cyan-400 group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute inset-0 w-32 h-32 rounded-full mx-auto bg-gradient-to-r from-purple-400/20 to-cyan-400/20 group-hover:animate-pulse"></div>
                </div>
                <h3 className="text-2xl font-bold text-white mb-2">{member.name}</h3>
                <p className="text-lg text-purple-300 font-semibold mb-4">{member.role}</p>
              </div>
              
              <p className="text-slate-400 leading-relaxed mb-6 text-center">
                {member.bio}
              </p>
              
              <div className="flex justify-center gap-4">
                <a
                  href={member.linkedin}
                  className="p-3 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl hover:from-blue-400/30 hover:to-blue-500/30 transition-all duration-300 group/link"
                >
                  <Linkedin className="w-5 h-5 text-blue-400 group-hover/link:scale-110 transition-transform" />
                </a>
                <a
                  href={member.github}
                  className="p-3 bg-gradient-to-br from-gray-500/20 to-gray-600/20 rounded-xl hover:from-gray-400/30 hover:to-gray-500/30 transition-all duration-300 group/link"
                >
                  <Github className="w-5 h-5 text-gray-400 group-hover/link:scale-110 transition-transform" />
                </a>
                <a
                  href={`mailto:${member.email}`}
                  className="p-3 bg-gradient-to-br from-purple-500/20 to-purple-600/20 rounded-xl hover:from-purple-400/30 hover:to-purple-500/30 transition-all duration-300 group/link"
                >
                  <Mail className="w-5 h-5 text-purple-400 group-hover/link:scale-110 transition-transform" />
                </a>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/40 backdrop-blur-xl rounded-3xl p-8 border border-slate-700/30 max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold text-white mb-6">Join Our Mission</h2>
            <p className="text-lg text-slate-400 leading-relaxed mb-8">
              We're always looking for talented individuals who share our passion for AI and innovation. 
              If you're interested in revolutionizing the insurance industry, we'd love to hear from you.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="mailto:careers@policyassistant.ai"
                className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-400 hover:to-pink-500 text-white px-8 py-4 rounded-2xl font-semibold text-lg transition-all duration-300 flex items-center gap-3 justify-center shadow-lg hover:shadow-purple-500/30 hover:shadow-2xl transform hover:scale-105"
              >
                <Mail className="w-5 h-5" />
                View Open Positions
              </a>
              <a
                href="mailto:hello@policyassistant.ai"
                className="px-8 py-4 rounded-2xl font-semibold text-lg border border-slate-600 text-slate-300 hover:border-purple-500/50 hover:text-white transition-all duration-300 flex items-center gap-3 justify-center"
              >
                <Users className="w-5 h-5" />
                Contact Us
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeamPage;