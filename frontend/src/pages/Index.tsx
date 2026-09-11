import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Icon } from "@/components/Icon";
import { BrandLink } from "@/components/BrandLink";
import logo from "@/assets/logo.png";

const Index = () => {
  return (
    <div className="min-h-screen">
      {/* Navigation */}
      <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <BrandLink />
          <div className="flex items-center gap-4">
            <Button asChild variant="ghost">
              <Link to="/login">Sign In</Link>
            </Button>
            <Button asChild>
              <Link to="/signup">Get Started</Link>
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-blue-100 via-purple-100 to-pink-100 dark:from-blue-950/40 dark:via-purple-950/40 dark:to-pink-950/40">
        {/* Animated Gradient Orbs */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/30 to-purple-400/30 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-purple-400/30 to-pink-400/30 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }}></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-br from-pink-400/20 to-blue-400/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "2s" }}></div>
        </div>

        <div className="container mx-auto px-4 py-20 lg:py-32 relative">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div className="space-y-4">
                <h1 className="text-4xl lg:text-6xl font-bold tracking-tight">
                  Your Health
                  <span className="text-primary block bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">Companion</span>
                </h1>
                <p className="text-xl text-muted-foreground max-w-lg leading-relaxed">
                  Maternal health monitoring with personalised insights, risk assessment, and
                  round-the-clock guidance through your pregnancy journey.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <Button asChild size="lg" className="text-lg px-8">
                  <Link to="/signup">
                    Start Your Journey
                    <Icon name="next" size={18} className="ml-2" />
                  </Link>
                </Button>
                <Button asChild variant="outline" size="lg" className="text-lg px-8">
                  <Link to="/login">Sign In</Link>
                </Button>
              </div>

              {/* What the service actually offers, in place of the usage and
                  accuracy figures that previously sat here. */}
              <div className="flex flex-wrap items-center gap-8 pt-4">
                <div>
                  <div className="text-2xl font-bold text-primary">6</div>
                  <div className="text-sm text-muted-foreground">Vitals tracked</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-primary">47</div>
                  <div className="text-sm text-muted-foreground">Counties covered</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-primary">2</div>
                  <div className="text-sm text-muted-foreground">Languages</div>
                </div>
              </div>
            </div>

            <div className="relative flex items-center justify-center">
              <div className="relative z-10 bg-gradient-to-br from-white via-blue-50/50 to-purple-50/50 dark:from-card dark:via-blue-950/20 dark:to-purple-950/20 rounded-3xl shadow-2xl p-16 border-2 border-primary/20 hover:border-primary/40 transition-all duration-300 backdrop-blur-sm">
                <div className="absolute inset-0 bg-gradient-to-tr from-primary/5 via-purple-500/5 to-pink-500/5 rounded-3xl"></div>
                <img src={logo} alt="AfyaJamii" className="h-48 w-48 mx-auto relative z-10" />
              </div>

              {/* Floating elements with animation */}
              <div className="absolute -top-4 -right-4 bg-gradient-to-br from-primary to-blue-600 text-primary-foreground p-4 rounded-full shadow-xl animate-bounce">
                <Icon name="chat" size={24} />
              </div>
              <div className="absolute -bottom-4 -left-4 bg-gradient-to-br from-green-500 to-emerald-600 text-white p-4 rounded-full shadow-xl animate-pulse">
                <Icon name="privacy" size={24} />
              </div>
              <div className="absolute top-1/2 -left-8 bg-gradient-to-br from-purple-500 to-pink-600 text-white p-3 rounded-full shadow-lg animate-pulse" style={{ animationDelay: "1s" }}>
                <Icon name="heartRate" size={20} />
              </div>
              <div className="absolute top-1/4 -right-8 bg-gradient-to-br from-blue-500 to-cyan-600 text-white p-3 rounded-full shadow-lg animate-bounce" style={{ animationDelay: "0.5s" }}>
                <Icon name="vitals" size={20} />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gradient-to-b from-background via-blue-50/30 to-background dark:from-background dark:via-blue-950/10 dark:to-background">
        <div className="container mx-auto px-4">
          <div className="text-center space-y-4 mb-16">
            <Badge variant="outline" className="mx-auto w-fit px-4 py-1.5">
              <Icon name="heartRate" size={13} className="mr-1.5" />
              Our Features
            </Badge>
            <h2 className="text-3xl lg:text-5xl font-bold">Comprehensive Maternal Care</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Everything you need for a healthy pregnancy and postnatal journey
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="relative overflow-hidden group hover:shadow-xl hover:scale-105 transition-all duration-300 border-2 hover:border-blue-200 dark:hover:border-blue-800">
              <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full -mr-16 -mt-16 group-hover:scale-150 transition-transform duration-500"></div>
              <CardHeader>
                <div className="h-14 w-14 bg-gradient-to-br from-blue-100 to-blue-200 dark:from-blue-900/20 dark:to-blue-800/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Icon name="vitals" size={28} className="text-blue-600 dark:text-blue-400" />
                </div>
                <CardTitle className="text-lg">Smart Vitals Monitoring</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  Track blood pressure, heart rate, temperature and more, with risk assessment
                  and clear alerts whenever something needs attention.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden group hover:shadow-xl hover:scale-105 transition-all duration-300 border-2 hover:border-green-200 dark:hover:border-green-800">
              <div className="absolute top-0 right-0 w-32 h-32 bg-green-500/5 rounded-full -mr-16 -mt-16 group-hover:scale-150 transition-transform duration-500"></div>
              <CardHeader>
                <div className="h-14 w-14 bg-gradient-to-br from-green-100 to-green-200 dark:from-green-900/20 dark:to-green-800/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Icon name="chat" size={28} className="text-green-600 dark:text-green-400" />
                </div>
                <CardTitle className="text-lg">Round-the-Clock Health Chat</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  Ask about symptoms, nutrition, or what a reading means, in English or Kiswahili,
                  whenever you need to ask.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden group hover:shadow-xl hover:scale-105 transition-all duration-300 border-2 hover:border-red-200 dark:hover:border-red-800">
              <div className="absolute top-0 right-0 w-32 h-32 bg-red-500/5 rounded-full -mr-16 -mt-16 group-hover:scale-150 transition-transform duration-500"></div>
              <CardHeader>
                <div className="h-14 w-14 bg-gradient-to-br from-red-100 to-red-200 dark:from-red-900/20 dark:to-red-800/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Icon name="privacy" size={28} className="text-red-600 dark:text-red-400" />
                </div>
                <CardTitle className="text-lg">Early Risk Detection</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  Your readings are checked against a model trained on maternal health records,
                  so concerns surface early rather than late.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden group hover:shadow-xl hover:scale-105 transition-all duration-300 border-2 hover:border-purple-200 dark:hover:border-purple-800">
              <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/5 rounded-full -mr-16 -mt-16 group-hover:scale-150 transition-transform duration-500"></div>
              <CardHeader>
                <div className="h-14 w-14 bg-gradient-to-br from-purple-100 to-purple-200 dark:from-purple-900/20 dark:to-purple-800/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Icon name="nutrition" size={28} className="text-purple-600 dark:text-purple-400" />
                </div>
                <CardTitle className="text-lg">Guidance Written for Here</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  Nutrition advice that names foods you can find at your local market, suited to
                  your stage of pregnancy.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 bg-gradient-to-br from-purple-50/50 via-pink-50/30 to-blue-50/50 dark:from-purple-950/10 dark:via-pink-950/5 dark:to-blue-950/10">
        <div className="container mx-auto px-4">
          <div className="text-center space-y-4 mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold">How AfyaJamii Works</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Simple steps to start tracking your health
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                title: "Sign Up & Setup",
                body: "Create your account and tell us whether you are pregnant, recently gave birth, or are here for general health.",
              },
              {
                title: "Track Your Vitals",
                body: "Record the numbers from your clinic card or home monitor and get an assessment straight away.",
              },
              {
                title: "Get Guidance",
                body: "Receive practical recommendations, and keep every reading in one place to show a health worker.",
              },
            ].map((step, index) => (
              <div key={step.title} className="text-center space-y-4">
                <div className="h-16 w-16 bg-primary text-primary-foreground rounded-full flex items-center justify-center mx-auto text-xl font-bold">
                  {index + 1}
                </div>
                <h3 className="text-xl font-semibold">{step.title}</h3>
                <p className="text-muted-foreground">{step.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20 bg-gradient-to-b from-background via-green-50/20 to-background dark:from-background dark:via-green-950/10 dark:to-background">
        <div className="container mx-auto px-4">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="space-y-8">
              <div className="space-y-4">
                <h2 className="text-3xl lg:text-4xl font-bold">Why Choose AfyaJamii?</h2>
                <p className="text-xl text-muted-foreground">
                  Careful technology, in service of compassionate maternal care
                </p>
              </div>

              <div className="space-y-6">
                <div className="flex gap-4">
                  <div className="h-8 w-8 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center flex-shrink-0">
                    <Icon name="check" size={18} className="text-green-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Built on Clinical Data</h3>
                    <p className="text-muted-foreground">Risk assessment comes from a model trained on maternal health records, not guesswork.</p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="h-8 w-8 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center flex-shrink-0">
                    <Icon name="accessibility" size={18} className="text-blue-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Easy to Use</h3>
                    <p className="text-muted-foreground">Adjustable text size, high contrast, and a clear layout that works on an inexpensive phone.</p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="h-8 w-8 bg-purple-100 dark:bg-purple-900/20 rounded-full flex items-center justify-center flex-shrink-0">
                    <Icon name="history" size={18} className="text-purple-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Your History, Kept</h3>
                    <p className="text-muted-foreground">Every reading and conversation stays in one place, so you can show how things have changed.</p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="h-8 w-8 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center flex-shrink-0">
                    <Icon name="privacy" size={18} className="text-red-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Privacy & Security</h3>
                    <p className="text-muted-foreground">Records are tied to your account and are never shown to other users of the service.</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="bg-gradient-to-br from-primary/10 to-purple-500/10 rounded-2xl p-8">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-card p-4 rounded-lg shadow-sm">
                    <Icon name="pregnant" size={32} className="text-primary mb-2" />
                    <div className="text-lg font-bold">Antenatal</div>
                    <div className="text-sm text-muted-foreground">Through pregnancy</div>
                  </div>
                  <div className="bg-white dark:bg-card p-4 rounded-lg shadow-sm">
                    <Icon name="postnatal" size={32} className="text-green-600 mb-2" />
                    <div className="text-lg font-bold">Postnatal</div>
                    <div className="text-sm text-muted-foreground">After the birth</div>
                  </div>
                  <div className="bg-white dark:bg-card p-4 rounded-lg shadow-sm">
                    <Icon name="nutrition" size={32} className="text-blue-600 mb-2" />
                    <div className="text-lg font-bold">Nutrition</div>
                    <div className="text-sm text-muted-foreground">Local foods</div>
                  </div>
                  <div className="bg-white dark:bg-card p-4 rounded-lg shadow-sm">
                    <Icon name="emergency" size={32} className="text-red-500 mb-2" />
                    <div className="text-lg font-bold">Emergency</div>
                    <div className="text-sm text-muted-foreground">Contacts by county</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t bg-gradient-to-b from-background to-slate-50 dark:from-background dark:to-slate-950 py-12">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="space-y-4">
              <BrandLink size={32} className="gap-2" nameClassName="text-lg" />
              <p className="text-muted-foreground">
                Maternal health monitoring and guidance for mothers in Kenya. Built to support
                the care you receive from a health worker, not to replace it.
              </p>
            </div>

            <div className="space-y-4">
              <h3 className="font-semibold flex items-center gap-2">
                <Icon name="emergency" size={16} className="text-red-500" />
                In an emergency
              </h3>
              <dl className="space-y-2 text-sm">
                {[
                  { label: "National emergency", number: "999" },
                  { label: "Kenya Red Cross ambulance", number: "1199" },
                  { label: "St John Ambulance", number: "0721 225 285" },
                ].map((item) => (
                  <div key={item.number} className="flex items-baseline justify-between gap-4">
                    <dt className="text-muted-foreground">{item.label}</dt>
                    <dd className="tabular font-medium">
                      <a href={`tel:${item.number.replace(/\s/g, "")}`} className="hover:underline">
                        {item.number}
                      </a>
                    </dd>
                  </div>
                ))}
              </dl>
              <p className="text-xs text-muted-foreground">
                Do not wait for advice from this app if someone is in danger. Call for help.
              </p>
            </div>

            <div className="space-y-4">
              <h3 className="font-semibold">Getting started</h3>
              <div className="space-y-2 text-sm">
                <Link to="/signup" className="block text-muted-foreground hover:text-foreground hover:underline">
                  Create an account
                </Link>
                <Link to="/login" className="block text-muted-foreground hover:text-foreground hover:underline">
                  Sign in
                </Link>
              </div>
            </div>
          </div>

          <div className="border-t mt-8 pt-8 text-center text-sm text-muted-foreground">
            <p>© {new Date().getFullYear()} AfyaJamii. Released under the MIT licence.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
